//! depth2depth: densify + denoise a metric depth image using an RGB frame.
//!
//! Depth Anything V2 (running on [candle](https://github.com/huggingface/candle))
//! predicts dense depth from RGB; that prediction is affine-fitted to the
//! trusted pixels of the raw sensor depth, then used to fill holes and replace
//! outliers. Raw depth is kept wherever it agrees with the aligned prediction,
//! so sensor geometry survives untouched.
//!
//! Model files: with the default `embedded-model` feature they are built into the library
//! (pinned by sha256 in model.json) and [`Depth2Depth::load`] needs nothing else. Without it,
//! see `tools/convert_weights.py` for converting the official
//! `depth_anything_v2_metric_hypersim_vits.pth` into the two safetensors files [`Depth2Depth::new`] loads.

pub mod calibrate;
pub mod cloud;
pub mod da2;
pub mod dinov2;
#[cfg(feature = "tensorrt")]
pub mod tensorrt;

pub use calibrate::{Anchor, Calibrated, Calibration, CalibrationConfig};
pub use cloud::{CloudOptions, Pinhole};

pub use candle;

/// The model files pinned in model.json, put in OUT_DIR by build.rs.
#[cfg(feature = "embedded-model")]
mod embedded {
    #[cfg(feature = "tensorrt")]
    pub static ONNX: &[u8] =
        include_bytes!(concat!(env!("OUT_DIR"), "/da2_metric_hypersim_vits_364x448.onnx"));
    #[cfg(feature = "tensorrt")]
    pub const ONNX_SHA256: &str = env!("D2D_SHA256_da2_metric_hypersim_vits_364x448_onnx");
    #[cfg(not(feature = "tensorrt"))]
    pub static DINOV2: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/dinov2_vits14.safetensors"));
    #[cfg(not(feature = "tensorrt"))]
    pub static HEAD: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/da2_head_vits.safetensors"));
}

/// Where [`Depth2Depth::load`] caches TensorRT engines: `$DEPTH2DEPTH_CACHE_DIR`, else
/// `$XDG_CACHE_HOME/depth2depth`, else `~/.cache/depth2depth`.
pub fn engine_cache_dir() -> std::path::PathBuf {
    use std::path::PathBuf;
    let env = |name: &str| std::env::var_os(name).filter(|value| !value.is_empty());
    env("DEPTH2DEPTH_CACHE_DIR")
        .map(PathBuf::from)
        .or_else(|| env("XDG_CACHE_HOME").map(|cache| PathBuf::from(cache).join("depth2depth")))
        .unwrap_or_else(|| {
            PathBuf::from(env("HOME").unwrap_or_else(|| ".".into())).join(".cache/depth2depth")
        })
}

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
    /// Fewer trusted pixels than this and the frame keeps the previous fit; a lidar
    /// cloud lands on far fewer pixels than a depth camera fills.
    pub min_fit_points: usize,
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
            min_fit_points: 500,
        }
    }
}

impl Config {
    /// Scale the model input resolution — the quality/speed knob. 1.0 keeps the
    /// current size, 0.5 is ~4x faster and coarser, 2.0 is ~4x slower and finer.
    /// Results snap to the 14-pixel patch grid (minimum 4 patches per side).
    pub fn with_quality(mut self, quality: f32) -> Self {
        let snap = |v: usize| ((v as f32 * quality / 14.0).round().max(4.0) as usize) * 14;
        self.model_h = snap(self.model_h);
        self.model_w = snap(self.model_w);
        self
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
    model: Model,
    config: Config,
    ema: Option<(f32, f32)>,
}

enum Model {
    Candle {
        network: da2::DepthAnythingV2,
        device: Device,
        dtype: DType,
    },
    #[cfg(feature = "tensorrt")]
    TensorRt(tensorrt::TrtDepth),
}

// Async module frameworks need the model to cross threads; keep it that way.
const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Depth2Depth>()
};

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
        let head_vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[head_safetensors], dtype, &device)? };
        Self::from_safetensors(dino_vb, head_vb, device, dtype, config)
    }

    fn from_safetensors(
        dino_vb: VarBuilder,
        head_vb: VarBuilder,
        device: Device,
        dtype: DType,
        config: Config,
    ) -> Result<Self> {
        let dino = dinov2::vit_small(dino_vb)?;
        let da2_config = da2::DepthAnythingV2Config::vit_small_metric(
            config.model_h,
            config.model_w,
            config.max_depth,
        );
        let network = da2::DepthAnythingV2::new(Arc::new(dino), da2_config, head_vb)?;
        Ok(Self {
            model: Model::Candle {
                network,
                device,
                dtype,
            },
            config,
            ema: None,
        })
    }

    /// The model as a TensorRT engine built from its ONNX export (see [`tensorrt`]); the
    /// model input size is the export's, whatever `config` says.
    #[cfg(feature = "tensorrt")]
    pub fn new_tensorrt(onnx_path: &str, engine_path: &str, config: Config) -> Result<Self> {
        let onnx = std::fs::read(onnx_path).map_err(|e| candle::Error::Msg(format!("{onnx_path}: {e}")))?;
        Self::tensorrt_from_onnx(&onnx, engine_path, config)
    }

    #[cfg(feature = "tensorrt")]
    fn tensorrt_from_onnx(onnx: &[u8], engine_path: &str, mut config: Config) -> Result<Self> {
        let engine = tensorrt::TrtDepth::open(onnx, engine_path).map_err(candle::Error::Msg)?;
        (config.model_h, config.model_w) = (engine.height, engine.width);
        Ok(Self {
            model: Model::TensorRt(engine),
            config,
            ema: None,
        })
    }

    /// The model built into the library (feature `embedded-model`), on the best backend compiled in:
    /// TensorRT with `tensorrt` (the engine is built on first use, minutes on an Orin, and cached
    /// under `engine_cache_dir()`), else candle on CUDA / Metal in f16 when available, else CPU in f32.
    #[cfg(feature = "embedded-model")]
    pub fn load(config: Config) -> Result<Self> {
        #[cfg(feature = "tensorrt")]
        {
            let directory = engine_cache_dir();
            std::fs::create_dir_all(&directory)
                .map_err(|e| candle::Error::Msg(format!("{}: {e}", directory.display())))?;
            let engine = directory.join(format!(
                "da2_metric_hypersim_vits_364x448_{}_trt{}.engine",
                &embedded::ONNX_SHA256[..12],
                tensorrt::version()
            ));
            Self::tensorrt_from_onnx(embedded::ONNX, &engine.to_string_lossy(), config)
        }
        #[cfg(not(feature = "tensorrt"))]
        {
            #[cfg(feature = "cuda")]
            let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
            #[cfg(all(feature = "metal", not(feature = "cuda")))]
            let device = Device::new_metal(0).unwrap_or(Device::Cpu);
            #[cfg(not(any(feature = "cuda", feature = "metal")))]
            let device = Device::Cpu;
            // candle's CPU f16 is ~3x slower than its f32.
            let dtype = if device.is_cpu() { DType::F32 } else { DType::F16 };
            Self::from_safetensors(
                VarBuilder::from_slice_safetensors(embedded::DINOV2, dtype, &device)?,
                VarBuilder::from_slice_safetensors(embedded::HEAD, dtype, &device)?,
                device,
                dtype,
                config,
            )
        }
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
        let pred_small: Vec<f32> = match &self.model {
            Model::Candle {
                network,
                device,
                dtype,
            } => {
                let input = Tensor::from_vec(chw, (1, 3, cfg.model_h, cfg.model_w), device)?
                    .to_dtype(*dtype)?;
                let depth = network.forward(&input)?;
                depth.to_dtype(DType::F32)?.flatten_all()?.to_vec1()?
            }
            #[cfg(feature = "tensorrt")]
            Model::TensorRt(engine) => engine.infer(&chw).map_err(candle::Error::Msg)?,
        };
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
        let fit = fit_affine(
            &pred,
            raw_depth_m,
            &valid,
            cfg.abs_tol,
            cfg.rel_tol,
            cfg.min_fit_points,
        );
        let (ema_a, ema_b) = match (self.ema, fit) {
            (None, Some(fit)) => fit,
            (None, None) => (1.0, 0.0),
            (Some(previous), None) => previous,
            (Some((pa, pb)), Some((a, b))) => {
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

    /// Dense metric depth for the rgb frame, calibrated to a sparse point cloud
    /// (camera frame, e.g. lidar moved into the camera's optical frame) rather than
    /// a depth image; `camera` is the rgb's intrinsics. Points hidden behind nearer
    /// ones from the camera's viewpoint are dropped first. See [`calibrate`].
    pub fn fuse_points(
        &mut self,
        rgb: &[u8],
        height: usize,
        width: usize,
        points: &[[f32; 3]],
        camera: &Pinhole,
        calibration: &mut Calibration,
    ) -> Result<Calibrated> {
        let pred = self.predict(rgb, height, width)?;
        let anchors = cloud::visible_anchors(points, camera, height, width);
        Ok(calibration.apply(&pred, height, width, &anchors))
    }
}

impl Fusion {
    /// The fused depth as camera-frame points, cropped and decimated by `options`.
    pub fn points(
        &self,
        height: usize,
        width: usize,
        camera: &Pinhole,
        options: &CloudOptions,
    ) -> Vec<[f32; 3]> {
        cloud::depth_to_points(&self.fused, height, width, camera, options)
    }
}

impl Calibrated {
    /// The calibrated depth as camera-frame points, cropped and decimated by `options`.
    pub fn points(
        &self,
        height: usize,
        width: usize,
        camera: &Pinhole,
        options: &CloudOptions,
    ) -> Vec<[f32; 3]> {
        cloud::depth_to_points(&self.depth, height, width, camera, options)
    }
}

/// Robust least-squares fit raw ~ a * pred + b over valid pixels; one refit
/// after dropping residual outliers. None when fewer than `min_points` take part.
pub fn fit_affine(
    pred: &[f32],
    raw: &[f32],
    valid: &[bool],
    abs_tol: f32,
    rel_tol: f32,
    min_points: usize,
) -> Option<(f32, f32)> {
    let mut fit = None;
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
        if n < min_points as f64 {
            break;
        }
        let det = spp * n - sp * sp;
        if det.abs() < 1e-9 {
            break;
        }
        let (af, bf) = (
            ((spr * n - sp * sr) / det) as f32,
            ((spp * sr - sp * spr) / det) as f32,
        );
        fit = Some((af, bf));
        for i in 0..pred.len() {
            let resid = (af * pred[i] + bf - raw[i]).abs();
            inlier[i] = valid[i] && resid < abs_tol.max(rel_tol * raw[i]);
        }
    }
    fit
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fit_recovers_the_scale_and_ignores_an_outlier() {
        let pred: Vec<f32> = (0..1000).map(|i| 1.0 + i as f32 * 0.005).collect();
        let mut raw: Vec<f32> = pred.iter().map(|p| 0.5 * p + 0.2).collect();
        raw[10] = 50.0;
        let valid = vec![true; raw.len()];
        let (a, b) = fit_affine(&pred, &raw, &valid, 0.3, 0.1, 500).unwrap();
        assert!((a - 0.5).abs() < 1e-3 && (b - 0.2).abs() < 1e-3, "{a} {b}");
    }

    #[test]
    fn too_few_anchor_points_give_no_fit() {
        let pred = vec![1.0f32, 2.0, 3.0];
        let raw = vec![2.0f32, 4.0, 6.0];
        assert_eq!(fit_affine(&pred, &raw, &[true; 3], 0.3, 0.1, 500), None);
        assert!(fit_affine(&pred, &raw, &[true; 3], 0.3, 0.1, 3).is_some());
    }
}
