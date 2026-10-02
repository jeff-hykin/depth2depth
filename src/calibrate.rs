//! Calibrating the model's depth to sparse metric anchors (a lidar scan, say).
//!
//! A single affine fit leaves the prediction's shape errors in place: on a robot's
//! head camera the floor near the bottom of the image comes out up to 2x too far
//! while the walls fit, which puts the near floor under the real one. Two
//! corrections fix that, blended per pixel:
//!
//! - **near anchors**, what the smooth fit below still gets wrong is spread edge-aware: each
//!   pixel averages its nearest anchors in (column, row, log predicted depth), so a
//!   correction measured on a wall does not leak onto the object in front of it;
//! - **away from anchors** (the near floor a lidar on the chassis never sees), a
//!   smooth fit `log z = g·log pred + c + quadratic(x, y)` carries the correction,
//!   its image-position part averaged over frames since it is mostly the lens.

use kiddo::{KdTree, SquaredEuclidean};
use rayon::prelude::*;

/// A pixel with a trusted metric depth.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub u: f32,
    pub v: f32,
    pub z: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CalibrationConfig {
    /// Edge-aware spread: one unit of distance is this many pixels...
    pub sigma_px: f32,
    /// ...or this much difference in log predicted depth.
    pub sigma_log_depth: f32,
    /// Anchors averaged per pixel.
    pub neighbours: usize,
    /// The edge-aware correction is solved every `grid_step` pixels and interpolated between.
    pub grid_step: usize,
    /// A pixel whose nearest anchor is under half this distance takes the edge-aware
    /// correction, past 1.5x it takes the smooth fit, and in between a blend.
    pub reach: f32,
    /// Weight of the newest frame in the average of the smooth fit's image-position part.
    pub shape_ema: f32,
    /// Fewer anchors than this and the frame keeps the previous smooth fit.
    pub min_anchors: usize,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            sigma_px: 40.0,
            sigma_log_depth: 0.15,
            neighbours: 16,
            grid_step: 4,
            reach: 1.0,
            shape_ema: 0.1,
            min_anchors: 100,
        }
    }
}

/// Depth after calibration, and how much of each pixel came from nearby anchors (0..1).
pub struct Calibrated {
    pub depth: Vec<f32>,
    pub support: Vec<f32>,
}

/// Calibration state carried between frames.
#[derive(Clone, Debug, Default)]
pub struct Calibration {
    pub config: CalibrationConfig,
    shape: Option<[f64; 5]>,
    scale: Option<(f64, f64)>,
}

impl Calibration {
    pub fn new(config: CalibrationConfig) -> Self {
        Self {
            config,
            shape: None,
            scale: None,
        }
    }

    /// Forget the averaged fit (scene cuts, recording seams).
    pub fn reset(&mut self) {
        self.shape = None;
        self.scale = None;
    }

    /// Calibrate `pred` (HxW meters) to `anchors`.
    pub fn apply(
        &mut self,
        pred: &[f32],
        height: usize,
        width: usize,
        anchors: &[Anchor],
    ) -> Calibrated {
        assert_eq!(pred.len(), height * width, "pred must be HxW");
        let anchors = in_image(anchors, height, width);
        let log_pred: Vec<f32> = pred.iter().map(|&p| p.max(1e-3).ln()).collect();
        if anchors.len() >= self.config.min_anchors {
            self.fit_smooth(&anchors, &log_pred, height, width);
        }
        let mut depth: Vec<f32> = (0..height * width)
            .into_par_iter()
            .map(|i| self.smooth(log_pred[i], i % width, i / width, height, width))
            .collect();
        let mut support = vec![0f32; height * width];
        if anchors.len() >= self.config.neighbours {
            let spread = Spread::new(self, &anchors, &log_pred, height, width);
            let step = self.config.grid_step.max(1);
            let (gh, gw) = ((height - 1) / step + 1, (width - 1) / step + 1);
            let (field, nearest): (Vec<f32>, Vec<f32>) = (0..gh * gw)
                .into_par_iter()
                .map(|cell| {
                    let (x, y) = ((cell % gw) * step, (cell / gw) * step);
                    spread.at(x, y, log_pred[y * width + x])
                })
                .unzip();
            let field = crate::bilinear_resize(&field, gh, gw, height, width);
            let nearest = crate::bilinear_resize(&nearest, gh, gw, height, width);
            depth
                .par_iter_mut()
                .zip(support.par_iter_mut())
                .zip(field.par_iter().zip(nearest.par_iter()))
                .for_each(|((depth, support), (field, nearest))| {
                    let w = self.weight(*nearest);
                    *depth *= (w * field).exp();
                    *support = w;
                });
        }
        Calibrated { depth, support }
    }

    /// `apply`, but only at the pixels `depth_to_points` reads under `options` (its crop and decimation;
    /// the rest stay 0): about 1/decimation² of the cost.
    pub fn apply_for_cloud(
        &mut self,
        pred: &[f32],
        height: usize,
        width: usize,
        anchors: &[Anchor],
        options: &crate::cloud::CloudOptions,
    ) -> Calibrated {
        assert_eq!(pred.len(), height * width, "pred must be HxW");
        let step = options.decimation.max(1);
        let [x0, y0, x1, y1] = options.roi.unwrap_or([0, 0, width, height]);
        let (x1, y1) = (x1.min(width), y1.min(height));
        let anchors = in_image(anchors, height, width);
        let log_pred: Vec<f32> = pred.iter().map(|&p| p.max(1e-3).ln()).collect();
        if anchors.len() >= self.config.min_anchors {
            self.fit_smooth(&anchors, &log_pred, height, width);
        }
        let spread = (anchors.len() >= self.config.neighbours)
            .then(|| Spread::new(self, &anchors, &log_pred, height, width));
        let (gh, gw) = (
            y1.saturating_sub(y0).div_ceil(step),
            x1.saturating_sub(x0).div_ceil(step),
        );
        let at = |cell: usize| (x0 + (cell % gw) * step, y0 + (cell / gw) * step);
        let values: Vec<(f32, f32)> = (0..gh * gw)
            .into_par_iter()
            .map(|cell| {
                let (x, y) = at(cell);
                let lp = log_pred[y * width + x];
                let smooth = self.smooth(lp, x, y, height, width);
                match &spread {
                    Some(spread) => {
                        let (field, nearest) = spread.at(x, y, lp);
                        let w = self.weight(nearest);
                        (smooth * (w * field).exp(), w)
                    }
                    None => (smooth, 0.0),
                }
            })
            .collect();
        let mut depth = vec![0f32; height * width];
        let mut support = vec![0f32; height * width];
        for (cell, (z, w)) in values.into_iter().enumerate() {
            let (x, y) = at(cell);
            let i = y * width + x;
            depth[i] = z;
            support[i] = w;
        }
        Calibrated { depth, support }
    }

    /// What `apply` would give at a few `(column, row)` pixels, without touching the averaged fit:
    /// for choosing extra anchors (a floor, say) before the real `apply`.
    pub fn probe(
        &self,
        pred: &[f32],
        height: usize,
        width: usize,
        anchors: &[Anchor],
        pixels: &[(usize, usize)],
    ) -> Vec<f32> {
        assert_eq!(pred.len(), height * width, "pred must be HxW");
        let mut fit = self.clone();
        let anchors = in_image(anchors, height, width);
        let log_pred: Vec<f32> = pred.iter().map(|&p| p.max(1e-3).ln()).collect();
        if anchors.len() >= fit.config.min_anchors {
            fit.fit_smooth(&anchors, &log_pred, height, width);
        }
        let spread = (anchors.len() >= fit.config.neighbours)
            .then(|| Spread::new(&fit, &anchors, &log_pred, height, width));
        pixels
            .par_iter()
            .map(|&(x, y)| {
                let lp = log_pred[y * width + x];
                let smooth = fit.smooth(lp, x, y, height, width);
                match &spread {
                    Some(spread) => {
                        let (field, nearest) = spread.at(x, y, lp);
                        smooth * (fit.weight(nearest) * field).exp()
                    }
                    None => smooth,
                }
            })
            .collect()
    }

    /// The smooth fit's depth at pixel (u, v).
    fn smooth(&self, log_pred: f32, u: usize, v: usize, height: usize, width: usize) -> f32 {
        let (g, c) = self.scale.unwrap_or((1.0, 0.0));
        let shape = self.shape.unwrap_or([0.0; 5]);
        (g * log_pred as f64 + c + dot(&shape, &quadratic(u, v, height, width))).exp() as f32
    }

    /// How much a pixel trusts the edge-aware correction, from its distance to the nearest anchor.
    fn weight(&self, nearest: f32) -> f32 {
        (1.5 - nearest / self.config.reach).clamp(0.0, 1.0)
    }

    /// Robust `log z = g·log pred + c + shape·quadratic(x, y)`; the shape is averaged over frames.
    fn fit_smooth(&mut self, anchors: &[Anchor], log_pred: &[f32], height: usize, width: usize) {
        let rows: Vec<([f64; 7], f64)> = anchors
            .iter()
            .map(|a| {
                let b = quadratic(a.u as usize, a.v as usize, height, width);
                let lp = log_pred[a.v as usize * width + a.u as usize] as f64;
                ([lp, 1.0, b[0], b[1], b[2], b[3], b[4]], (a.z as f64).ln())
            })
            .collect();
        let Some(full) = robust_lstsq::<7>(&rows) else {
            return;
        };
        let fresh: [f64; 5] = full[2..].try_into().unwrap();
        let shape = match self.shape {
            None => fresh,
            Some(old) => {
                let k = self.config.shape_ema as f64;
                std::array::from_fn(|i| (1.0 - k) * old[i] + k * fresh[i])
            }
        };
        self.shape = Some(shape);
        let rows: Vec<([f64; 2], f64)> = rows
            .iter()
            .map(|(x, y)| ([x[0], 1.0], y - dot(&shape, &x[2..].try_into().unwrap())))
            .collect();
        if let Some([g, c]) = robust_lstsq::<2>(&rows) {
            self.scale = Some((g, c));
        }
    }
}

/// The anchors that land inside the image with a positive depth.
fn in_image(anchors: &[Anchor], height: usize, width: usize) -> Vec<Anchor> {
    anchors
        .iter()
        .copied()
        .filter(|a| {
            a.z > 0.0
                && a.u >= 0.0
                && a.v >= 0.0
                && (a.u as usize) < width
                && (a.v as usize) < height
        })
        .collect()
}

/// What the smooth fit still gets wrong at each anchor, spread edge-aware: a pixel averages its
/// nearest anchors in (column, row, log predicted depth).
struct Spread<'a> {
    config: &'a CalibrationConfig,
    tree: KdTree<f32, 3>,
    residual: Vec<f32>,
}

impl<'a> Spread<'a> {
    fn new(
        fit: &'a Calibration,
        anchors: &[Anchor],
        log_pred: &[f32],
        height: usize,
        width: usize,
    ) -> Self {
        let mut spread = Spread {
            config: &fit.config,
            tree: KdTree::new(),
            residual: Vec::with_capacity(anchors.len()),
        };
        for (i, a) in anchors.iter().enumerate() {
            let (u, v) = (a.u as usize, a.v as usize);
            let lp = log_pred[v * width + u];
            spread.tree.add(&spread.feature(a.u, a.v, lp), i as u64);
            spread
                .residual
                .push(a.z.ln() - fit.smooth(lp, u, v, height, width).ln());
        }
        spread
    }

    fn feature(&self, u: f32, v: f32, log_pred: f32) -> [f32; 3] {
        let cfg = self.config;
        [
            u / cfg.sigma_px,
            v / cfg.sigma_px,
            log_pred / cfg.sigma_log_depth,
        ]
    }

    /// The averaged residual (log) at a pixel, and its distance to the nearest anchor.
    fn at(&self, x: usize, y: usize, log_pred: f32) -> (f32, f32) {
        let found = self.tree.nearest_n::<SquaredEuclidean>(
            &self.feature(x as f32, y as f32, log_pred),
            self.config.neighbours,
        );
        let (mut sum, mut weights) = (0f32, 0f32);
        for n in &found {
            let w = (-0.5 * n.distance).exp() + 1e-9;
            sum += w * self.residual[n.item as usize];
            weights += w;
        }
        let nearest = found.first().map_or(f32::INFINITY, |n| n.distance.sqrt());
        (sum / weights, nearest)
    }
}

/// `[x, y, x², xy, y²]` with x, y the offset from the image centre in image widths / heights.
fn quadratic(u: usize, v: usize, height: usize, width: usize) -> [f64; 5] {
    let x = (u as f64 - (width as f64 - 1.0) / 2.0) / width as f64;
    let y = (v as f64 - (height as f64 - 1.0) / 2.0) / height as f64;
    [x, y, x * x, x * y, y * y]
}

fn dot(a: &[f64; 5], b: &[f64; 5]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

/// Least squares, refit three times on the rows whose residual is under
/// max(0.1, its 70th percentile); None when too few rows or a singular system.
fn robust_lstsq<const N: usize>(rows: &[([f64; N], f64)]) -> Option<[f64; N]> {
    let mut keep = vec![true; rows.len()];
    let mut solution = None;
    for _ in 0..4 {
        let kept: Vec<_> = rows
            .iter()
            .zip(&keep)
            .filter(|(_, k)| **k)
            .map(|(r, _)| r)
            .collect();
        if kept.len() < 3 * N {
            break;
        }
        let Some(x) = solve_normal::<N>(&kept) else {
            break;
        };
        let residuals: Vec<f64> = rows
            .iter()
            .map(|(a, y)| (a.iter().zip(&x).map(|(a, x)| a * x).sum::<f64>() - y).abs())
            .collect();
        let mut sorted = residuals.clone();
        sorted.sort_by(f64::total_cmp);
        let cut = sorted[(sorted.len() * 7 / 10).min(sorted.len() - 1)].max(0.1);
        keep = residuals.iter().map(|r| *r < cut).collect();
        solution = Some(x);
    }
    solution
}

fn solve_normal<const N: usize>(rows: &[&([f64; N], f64)]) -> Option<[f64; N]> {
    let mut m = [[0f64; N]; N];
    let mut rhs = [0f64; N];
    for (a, y) in rows {
        for i in 0..N {
            rhs[i] += a[i] * y;
            for j in 0..N {
                m[i][j] += a[i] * a[j];
            }
        }
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..N {
        let pivot = (col..N).max_by(|&a, &b| m[a][col].abs().total_cmp(&m[b][col].abs()))?;
        if m[pivot][col].abs() < 1e-12 {
            return None;
        }
        m.swap(col, pivot);
        rhs.swap(col, pivot);
        for row in col + 1..N {
            let f = m[row][col] / m[col][col];
            for k in col..N {
                m[row][k] -= f * m[col][k];
            }
            rhs[row] -= f * rhs[col];
        }
    }
    let mut x = [0f64; N];
    for row in (0..N).rev() {
        let tail: f64 = (row + 1..N).map(|k| m[row][k] * x[k]).sum();
        x[row] = (rhs[row] - tail) / m[row][row];
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: usize = 60;
    const W: usize = 80;

    fn grid_anchors(truth: &dyn Fn(usize, usize) -> f32, every: usize) -> Vec<Anchor> {
        (0..H)
            .step_by(every)
            .flat_map(|v| (0..W).step_by(every).map(move |u| (u, v)))
            .map(|(u, v)| Anchor {
                u: u as f32,
                v: v as f32,
                z: truth(u, v),
            })
            .collect()
    }

    #[test]
    fn a_prediction_off_by_a_power_and_a_scale_is_recovered() {
        let truth = |u: usize, v: usize| 1.0 + (u + v) as f32 * 0.05;
        let pred: Vec<f32> = (0..H * W)
            .map(|i| 2.0 * truth(i % W, i / W).powf(0.8))
            .collect();
        let mut calibration = Calibration::new(CalibrationConfig::default());
        let out = calibration.apply(&pred, H, W, &grid_anchors(&truth, 3));
        for i in (0..H * W).step_by(97) {
            let want = truth(i % W, i / W);
            assert!(
                (out.depth[i] - want).abs() < 0.02 * want,
                "pixel {i}: {} vs {want}",
                out.depth[i]
            );
        }
    }

    #[test]
    fn a_correction_on_the_wall_does_not_leak_onto_the_object_in_front() {
        // Left half a wall at 5 m the model reads 2x too far, right half an object at 1 m it reads right;
        // anchors only on the wall, and along the boundary on the object.
        let object = |u: usize| u >= W / 2;
        let truth = |u: usize, _v: usize| if object(u) { 1.0 } else { 5.0 };
        let pred: Vec<f32> = (0..H * W)
            .map(|i| if object(i % W) { 1.0 } else { 10.0 })
            .collect();
        let anchors: Vec<Anchor> = grid_anchors(&truth, 2)
            .into_iter()
            .filter(|a| !object(a.u as usize) || (a.u as usize) < W / 2 + 4)
            .collect();
        let config = CalibrationConfig {
            min_anchors: usize::MAX,
            ..Default::default()
        };
        let out = Calibration::new(config).apply(&pred, H, W, &anchors);
        let deep_in_object = (H / 2) * W + W - 2;
        assert!(
            (out.depth[deep_in_object] - 1.0).abs() < 0.05,
            "{}",
            out.depth[deep_in_object]
        );
        assert!((out.depth[(H / 2) * W + 2] - 5.0).abs() < 0.1);
    }

    #[test]
    fn far_from_every_anchor_the_smooth_fit_takes_over() {
        let truth = |_u: usize, v: usize| 1.0 + v as f32 * 0.1;
        let pred: Vec<f32> = (0..H * W).map(|i| 1.5 * truth(i % W, i / W)).collect();
        // Anchors only in the top half; the bottom rows have none nearby.
        let anchors: Vec<Anchor> = grid_anchors(&truth, 2)
            .into_iter()
            .filter(|a| a.v < 20.0)
            .collect();
        let out = Calibration::new(CalibrationConfig::default()).apply(&pred, H, W, &anchors);
        let bottom = (H - 1) * W + W / 2;
        assert_eq!(out.support[bottom], 0.0);
        assert!(out.support[5 * W + W / 2] > 0.9);
        assert!(
            (out.depth[bottom] - truth(0, H - 1)).abs() < 0.05 * truth(0, H - 1),
            "{}",
            out.depth[bottom]
        );
    }

    #[test]
    fn without_enough_anchors_the_previous_fit_is_kept() {
        let truth = |u: usize, v: usize| 2.0 + (u + v) as f32 * 0.02;
        let pred: Vec<f32> = (0..H * W).map(|i| 3.0 * truth(i % W, i / W)).collect();
        let mut calibration = Calibration::new(CalibrationConfig::default());
        let first = calibration.apply(&pred, H, W, &grid_anchors(&truth, 2));
        let second = calibration.apply(&pred, H, W, &[]);
        assert!(
            (first.depth[123] - second.depth[123]).abs() < 0.05,
            "{} {}",
            first.depth[123],
            second.depth[123]
        );
    }

    #[test]
    fn a_probe_matches_apply_and_leaves_the_averaged_fit_alone() {
        let truth = |u: usize, v: usize| 1.0 + (u + v) as f32 * 0.05;
        let pred: Vec<f32> = (0..H * W)
            .map(|i| 2.0 * truth(i % W, i / W).powf(0.8))
            .collect();
        let anchors = grid_anchors(&truth, 3);
        let calibration = Calibration::new(CalibrationConfig::default());
        let probed = calibration.probe(&pred, H, W, &anchors, &[(40, 30), (8, 52)]);
        assert!(calibration.shape.is_none() && calibration.scale.is_none());
        let applied = calibration.clone().apply(&pred, H, W, &anchors);
        for (p, (x, y)) in probed.iter().zip([(40, 30), (8, 52)]) {
            assert!(
                (p - applied.depth[y * W + x]).abs() < 0.02 * p,
                "{p} {}",
                applied.depth[y * W + x]
            );
        }
    }

    #[test]
    fn apply_for_cloud_gives_the_cloud_of_a_full_apply() {
        let truth = |u: usize, v: usize| 1.0 + (u + v) as f32 * 0.05;
        let pred: Vec<f32> = (0..H * W)
            .map(|i| 2.0 * truth(i % W, i / W).powf(0.8))
            .collect();
        let anchors = grid_anchors(&truth, 3);
        let camera = crate::cloud::Pinhole {
            fx: 60.0,
            fy: 60.0,
            cx: 40.0,
            cy: 30.0,
        };
        let calibration = Calibration::new(CalibrationConfig::default());
        let full = calibration.clone().apply(&pred, H, W, &anchors);
        for roi in [None, Some([3, 5, 71, 50])] {
            let options = crate::cloud::CloudOptions {
                decimation: 8,
                roi,
                ..Default::default()
            };
            let sparse = calibration
                .clone()
                .apply_for_cloud(&pred, H, W, &anchors, &options);
            let kept = sparse.depth.iter().filter(|&&z| z > 0.0).count();
            let from_full = full.points(H, W, &camera, &options);
            let from_sparse = sparse.points(H, W, &camera, &options);
            assert_eq!(from_sparse.len(), from_full.len());
            assert_eq!(
                from_sparse.len(),
                kept,
                "every calibrated pixel becomes a point"
            );
            for (a, b) in from_sparse.iter().zip(&from_full) {
                assert!((a[2] - b[2]).abs() < 0.02 * b[2], "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn the_robust_fit_ignores_gross_outliers() {
        let rows: Vec<([f64; 2], f64)> = (0..100)
            .map(|i| {
                let x = i as f64 * 0.1;
                ([x, 1.0], if i % 10 == 0 { 100.0 } else { 2.0 * x + 0.5 })
            })
            .collect();
        let [a, b] = robust_lstsq::<2>(&rows).unwrap();
        assert!((a - 2.0).abs() < 1e-6 && (b - 0.5).abs() < 1e-6, "{a} {b}");
    }
}
