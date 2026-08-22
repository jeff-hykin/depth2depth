//! depth2depth: densify + denoise a metric depth image using an RGB frame.
//!
//! Depth Anything V2 (running on [candle](https://github.com/huggingface/candle))
//! predicts dense depth from RGB; that prediction is affine-fitted to the
//! trusted pixels of the raw sensor depth, then used to fill holes and replace
//! outliers. Raw depth is kept wherever it agrees with the aligned prediction,
//! so sensor geometry survives untouched.
//!
//! Model files: see `tools/convert_weights.py` for converting the official
//! `depth_anything_v2_metric_hypersim_vits.pth` into the two safetensors files
//! this crate loads.

pub mod da2;
pub mod dinov2;

pub use candle;

use std::sync::Arc;

use candle::{DType, Device, Module, Result, Tensor};
use candle_nn::VarBuilder;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

#[derive(Clone, Debug)]
pub struct Config {
    /// Model input size; both must be multiples of 14. Smaller = faster.
    pub model_h: usize,
    pub model_w: usize,
    /// Metric head range (Hypersim indoor checkpoint uses 20m).
    pub max_depth: f64,
    /// Raw depth trust range in meters; outside = hole to fill.
    pub near_m: f32,
    pub far_m: f32,
    /// Weight of the newest affine fit in the EMA (1.0 = no smoothing).
    pub ema_new_weight: f32,
    /// A raw pixel is kept when |aligned - raw| < max(abs_tol, rel_tol * aligned).
    pub abs_tol: f32,
    pub rel_tol: f32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            model_h: 280,
            model_w: 504,
            max_depth: 20.0,
            near_m: 0.3,
            far_m: 6.0,
            ema_new_weight: 0.3,
            abs_tol: 0.3,
            rel_tol: 0.1,
        }
    }
}

pub struct Fusion {
    /// Dense metric depth, meters, same resolution as the input.
    pub fused: Vec<f32>,
    /// The affine-aligned model prediction alone.
    pub aligned: Vec<f32>,
    /// Per-pixel: true where the raw sensor value was kept.
    pub kept_raw: Vec<bool>,
    /// Smoothed affine parameters raw ~ a * prediction + b.
    pub a: f32,
    pub b: f32,
}

pub struct Depth2Depth {
    model: da2::DepthAnythingV2,
    device: Device,
    dtype: DType,
    config: Config,
    ema: Option<(f32, f32)>,
}

impl Depth2Depth {
    pub fn new(
        dinov2_safetensors: &str,
        head_safetensors: &str,
        device: Device,
        dtype: DType,
        config: Config,
    ) -> Result<Self> {
        let dino_vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dinov2_safetensors], dtype, &device)?
        };
        let dino = dinov2::vit_small(dino_vb)?;
        let head_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[head_safetensors], dtype, &device)? };
        let da2_config = da2::DepthAnythingV2Config::vit_small_metric(
            config.model_h,
            config.model_w,
            config.max_depth,
        );
        let model = da2::DepthAnythingV2::new(Arc::new(dino), da2_config, head_vb)?;
        Ok(Self {
            model,
            device,
            dtype,
            config,
            ema: None,
        })
    }

    /// Reset the temporal affine smoothing (call on scene cuts / recording seams).
    pub fn reset(&mut self) {
        self.ema = None;
    }

    /// Model-only depth prediction in meters at (height, width) resolution.
    pub fn predict(&self, rgb: &[u8], height: usize, width: usize) -> Result<Vec<f32>> {
        let cfg = &self.config;
        assert_eq!(rgb.len(), 3 * height * width, "rgb must be HxWx3 u8");
        let mut chw = vec![0f32; 3 * cfg.model_h * cfg.model_w];
        for channel in 0..3 {
            let plane: Vec<f32> = (0..height * width)
                .map(|i| rgb[3 * i + channel] as f32)
                .collect();
            let small = bilinear_resize(&plane, height, width, cfg.model_h, cfg.model_w);
            for (i, v) in small.iter().enumerate() {
                chw[channel * cfg.model_h * cfg.model_w + i] =
                    (v / 255.0 - IMAGENET_MEAN[channel]) / IMAGENET_STD[channel];
            }
        }
        let input = Tensor::from_vec(chw, (1, 3, cfg.model_h, cfg.model_w), &self.device)?
            .to_dtype(self.dtype)?;
        let depth = self.model.forward(&input)?;
        let pred_small: Vec<f32> = depth.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?;
        Ok(bilinear_resize(
            &pred_small,
            cfg.model_h,
            cfg.model_w,
            height,
            width,
        ))
    }

    /// Fuse a raw metric depth image (meters, same resolution as rgb, 0 or
    /// out-of-range = hole) with the model prediction for the rgb frame.
    pub fn fuse(
        &mut self,
        rgb: &[u8],
        raw_depth_m: &[f32],
        height: usize,
        width: usize,
    ) -> Result<Fusion> {
        assert_eq!(raw_depth_m.len(), height * width);
        let cfg = self.config.clone();
        let pred = self.predict(rgb, height, width)?;

        let valid: Vec<bool> = raw_depth_m
            .iter()
            .map(|&z| (cfg.near_m..=cfg.far_m).contains(&z))
            .collect();
        let (a, b) = fit_affine(&pred, raw_depth_m, &valid, cfg.abs_tol, cfg.rel_tol);
        let (ema_a, ema_b) = match self.ema {
            None => (a, b),
            Some((pa, pb)) => {
                let k = cfg.ema_new_weight;
                ((1.0 - k) * pa + k * a, (1.0 - k) * pb + k * b)
            }
        };
        self.ema = Some((ema_a, ema_b));

        let aligned: Vec<f32> = pred.iter().map(|&p| ema_a * p + ema_b).collect();
        let mut fused = vec![0f32; raw_depth_m.len()];
        let mut kept_raw = vec![false; raw_depth_m.len()];
        for i in 0..raw_depth_m.len() {
            let keep = valid[i]
                && (aligned[i] - raw_depth_m[i]).abs()
                    < cfg.abs_tol.max(cfg.rel_tol * aligned[i]);
            kept_raw[i] = keep;
            fused[i] = if keep { raw_depth_m[i] } else { aligned[i] };
        }
        Ok(Fusion {
            fused,
            aligned,
            kept_raw,
            a: ema_a,
            b: ema_b,
        })
    }
}

/// Robust least-squares fit raw ~ a * pred + b over valid pixels; one refit
/// after dropping residual outliers.
pub fn fit_affine(
    pred: &[f32],
    raw: &[f32],
    valid: &[bool],
    abs_tol: f32,
    rel_tol: f32,
) -> (f32, f32) {
    let mut a = 1.0f64;
    let mut b = 0.0f64;
    let mut inlier: Vec<bool> = valid.to_vec();
    for _ in 0..2 {
        let (mut sp, mut spp, mut sr, mut spr, mut n) = (0f64, 0f64, 0f64, 0f64, 0f64);
        for i in 0..pred.len() {
            if inlier[i] {
                let p = pred[i] as f64;
                let r = raw[i] as f64;
                sp += p;
                spp += p * p;
                sr += r;
                spr += p * r;
                n += 1.0;
            }
        }
        if n < 500.0 {
            break;
        }
        let det = spp * n - sp * sp;
        if det.abs() < 1e-9 {
            break;
        }
        a = (spr * n - sp * sr) / det;
        b = (spp * sr - sp * spr) / det;
        let (af, bf) = (a as f32, b as f32);
        for i in 0..pred.len() {
            let resid = (af * pred[i] + bf - raw[i]).abs();
            inlier[i] = valid[i] && resid < abs_tol.max(rel_tol * raw[i]);
        }
    }
    (a as f32, b as f32)
}

pub fn bilinear_resize(
    src: &[f32],
    src_h: usize,
    src_w: usize,
    dst_h: usize,
    dst_w: usize,
) -> Vec<f32> {
    let mut dst = vec![0f32; dst_h * dst_w];
    let scale_y = src_h as f32 / dst_h as f32;
    let scale_x = src_w as f32 / dst_w as f32;
    for y in 0..dst_h {
        let sy = ((y as f32 + 0.5) * scale_y - 0.5).clamp(0.0, (src_h - 1) as f32);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(src_h - 1);
        let fy = sy - y0 as f32;
        for x in 0..dst_w {
            let sx = ((x as f32 + 0.5) * scale_x - 0.5).clamp(0.0, (src_w - 1) as f32);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(src_w - 1);
            let fx = sx - x0 as f32;
            let top = src[y0 * src_w + x0] * (1.0 - fx) + src[y0 * src_w + x1] * fx;
            let bot = src[y1 * src_w + x0] * (1.0 - fx) + src[y1 * src_w + x1] * fx;
            dst[y * dst_w + x] = top * (1.0 - fy) + bot * fy;
        }
    }
    dst
}
