//! Point clouds in and out: a sparse sensor cloud (lidar) as the anchor instead of a
//! depth image, and the fused depth back out as a cropped, decimated cloud.
//!
//! Points are in the camera's optical frame: x right, y down, z forward, meters.

use crate::calibrate::Anchor;

/// Intrinsics of an undistorted (pinhole) image.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pinhole {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
}

impl Pinhole {
    /// The same camera at `scale` times the resolution, e.g. 0.5 for an image decoded at half size.
    pub fn scaled(&self, scale: f32) -> Self {
        Self {
            fx: self.fx * scale,
            fy: self.fy * scale,
            cx: (self.cx + 0.5) * scale - 0.5,
            cy: (self.cy + 0.5) * scale - 0.5,
        }
    }
}

/// Splat points into a depth image (meters, 0 = no reading); the nearest point wins a pixel.
pub fn points_to_depth(
    points: &[[f32; 3]],
    camera: &Pinhole,
    height: usize,
    width: usize,
) -> Vec<f32> {
    let mut depth = vec![0f32; height * width];
    for &[x, y, z] in points {
        if !(z > 0.0) {
            continue;
        }
        let u = (camera.fx * x / z + camera.cx).round();
        let v = (camera.fy * y / z + camera.cy).round();
        if u < 0.0 || v < 0.0 || u >= width as f32 || v >= height as f32 {
            continue;
        }
        let pixel = &mut depth[v as usize * width + u as usize];
        if *pixel == 0.0 || z < *pixel {
            *pixel = z;
        }
    }
    depth
}

/// The points as depth anchors, minus those hidden from the camera: a sensor mounted
/// elsewhere sees past the edge of an object the camera sees, and those far points
/// land on the object's pixels. A point is dropped when a point within
/// `OCCLUSION_WINDOW` pixels is more than 10% (+0.1 m) nearer.
pub fn visible_anchors(
    points: &[[f32; 3]],
    camera: &Pinhole,
    height: usize,
    width: usize,
) -> Vec<Anchor> {
    const OCCLUSION_WINDOW: usize = 9;
    let depth = points_to_depth(points, camera, height, width);
    let half = OCCLUSION_WINDOW / 2;
    let mut anchors = Vec::new();
    for v in 0..height {
        for u in 0..width {
            let z = depth[v * width + u];
            if z == 0.0 {
                continue;
            }
            let nearest = (v.saturating_sub(half)..(v + half + 1).min(height))
                .flat_map(|y| {
                    (u.saturating_sub(half)..(u + half + 1).min(width)).map(move |x| (x, y))
                })
                .map(|(x, y)| depth[y * width + x])
                .filter(|&d| d > 0.0)
                .fold(f32::INFINITY, f32::min);
            if z <= nearest * 1.1 + 0.1 {
                anchors.push(Anchor {
                    u: u as f32,
                    v: v as f32,
                    z,
                });
            }
        }
    }
    anchors
}

/// The nearest anchor in each `step` x `step` pixel block (1 keeps them all): calibration time
/// follows the anchor count, and a dense lidar history gives far more than the fit needs.
pub fn thin_anchors(anchors: Vec<Anchor>, step: usize, height: usize, width: usize) -> Vec<Anchor> {
    if step <= 1 {
        return anchors;
    }
    let columns = width.div_ceil(step);
    let mut nearest: Vec<Option<Anchor>> = vec![None; height.div_ceil(step) * columns];
    for anchor in anchors {
        let held = &mut nearest[(anchor.v as usize / step) * columns + anchor.u as usize / step];
        if held.is_none_or(|held| anchor.z < held.z) {
            *held = Some(anchor);
        }
    }
    nearest.into_iter().flatten().collect()
}

/// Which pixels become points, and how many. Every crop runs before any decimation,
/// so the point budget is spent only on what survives the crop.
#[derive(Clone, Debug, PartialEq)]
pub struct CloudOptions {
    /// Keep points whose distance from the camera lies in this range, meters.
    pub min_range_m: f32,
    pub max_range_m: f32,
    /// Pixel window `[x0, y0, x1, y1)`; None keeps the whole image.
    pub roi: Option<[usize; 4]>,
    /// Camera-frame box `(min, max)`; None keeps everything.
    pub bounds: Option<([f32; 3], [f32; 3])>,
    /// Keep one pixel in every `decimation` x `decimation` block of the crop (1 keeps all).
    pub decimation: usize,
    /// After that, thin evenly down to at most this many points; None keeps them all.
    pub max_points: Option<usize>,
}

impl Default for CloudOptions {
    fn default() -> Self {
        Self {
            min_range_m: 0.0,
            max_range_m: f32::INFINITY,
            roi: None,
            bounds: None,
            decimation: 1,
            max_points: None,
        }
    }
}

/// Unproject a depth image (meters, 0 or non-finite = none) into camera-frame points.
pub fn depth_to_points(
    depth: &[f32],
    height: usize,
    width: usize,
    camera: &Pinhole,
    options: &CloudOptions,
) -> Vec<[f32; 3]> {
    assert_eq!(depth.len(), height * width, "depth must be HxW");
    let [x0, y0, x1, y1] = options.roi.unwrap_or([0, 0, width, height]);
    let (x1, y1) = (x1.min(width), y1.min(height));
    let step = options.decimation.max(1);
    let mut points = Vec::new();
    for v in (y0..y1).step_by(step) {
        for u in (x0..x1).step_by(step) {
            let z = depth[v * width + u];
            if !(z > 0.0 && z.is_finite()) {
                continue;
            }
            let point = [
                (u as f32 - camera.cx) * z / camera.fx,
                (v as f32 - camera.cy) * z / camera.fy,
                z,
            ];
            let range = (point[0] * point[0] + point[1] * point[1] + z * z).sqrt();
            if range < options.min_range_m || range > options.max_range_m {
                continue;
            }
            if let Some((low, high)) = options.bounds {
                if (0..3).any(|axis| point[axis] < low[axis] || point[axis] > high[axis]) {
                    continue;
                }
            }
            points.push(point);
        }
    }
    match options.max_points {
        Some(budget) if points.len() > budget => thin(points, budget),
        _ => points,
    }
}

/// Evenly spaced picks, so the kept points still cover the whole crop.
fn thin(points: Vec<[f32; 3]>, budget: usize) -> Vec<[f32; 3]> {
    let stride = points.len() as f64 / budget as f64;
    (0..budget)
        .map(|i| points[(i as f64 * stride) as usize])
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thinning_keeps_the_nearest_anchor_of_each_block() {
        let anchor = |u: f32, v: f32, z: f32| Anchor { u, v, z };
        let anchors = vec![
            anchor(0.0, 0.0, 3.0),
            anchor(3.0, 3.0, 2.0),
            anchor(4.0, 0.0, 5.0),
        ];
        assert_eq!(
            thin_anchors(anchors.clone(), 4, 8, 8),
            vec![anchor(3.0, 3.0, 2.0), anchor(4.0, 0.0, 5.0)]
        );
        assert_eq!(thin_anchors(anchors.clone(), 1, 8, 8), anchors);
    }

    const CAMERA: Pinhole = Pinhole {
        fx: 100.0,
        fy: 100.0,
        cx: 50.0,
        cy: 40.0,
    };

    #[test]
    fn a_point_lands_on_its_pixel_and_the_nearest_wins() {
        let depth = points_to_depth(&[[0.1, 0.0, 2.0], [0.2, 0.0, 4.0]], &CAMERA, 80, 100);
        assert_eq!(depth[40 * 100 + 55], 2.0);
        let hidden = points_to_depth(&[[0.2, 0.0, 4.0], [0.1, 0.0, 2.0]], &CAMERA, 80, 100);
        assert_eq!(hidden[40 * 100 + 55], 2.0);
    }

    #[test]
    fn points_behind_or_outside_the_image_are_dropped() {
        let depth = points_to_depth(&[[0.0, 0.0, -1.0], [10.0, 0.0, 1.0]], &CAMERA, 80, 100);
        assert!(depth.iter().all(|&z| z == 0.0));
    }

    #[test]
    fn unprojecting_a_splatted_cloud_returns_the_points() {
        let points = [[0.3, -0.2, 2.0], [-0.5, 0.25, 5.0]];
        let depth = points_to_depth(&points, &CAMERA, 80, 100);
        let back = depth_to_points(&depth, 80, 100, &CAMERA, &CloudOptions::default());
        assert_eq!(back.len(), 2);
        for (a, b) in back.iter().zip(points) {
            assert!((0..3).all(|i| (a[i] - b[i]).abs() < 1e-5), "{a:?} vs {b:?}");
        }
    }

    #[test]
    fn range_and_box_crops_drop_points() {
        let depth = vec![3.0f32; 80 * 100];
        let ranged = CloudOptions {
            max_range_m: 3.05,
            ..Default::default()
        };
        let near = depth_to_points(&depth, 80, 100, &CAMERA, &ranged);
        assert!(!near.is_empty() && near.len() < 8000);
        let boxed = CloudOptions {
            bounds: Some(([-10.0, 0.0, 0.0], [10.0, 10.0, 10.0])),
            ..Default::default()
        };
        assert!(depth_to_points(&depth, 80, 100, &CAMERA, &boxed)
            .iter()
            .all(|p| p[1] >= 0.0));
    }

    #[test]
    fn the_point_budget_is_spent_after_the_crop() {
        // Crop first: the whole budget lands inside the window. Thinning the
        // full image first would leave only the fraction that happens to fall in it.
        let depth = vec![2.0f32; 80 * 100];
        let options = CloudOptions {
            roi: Some([0, 0, 10, 10]),
            max_points: Some(50),
            ..Default::default()
        };
        let cropped_then_thinned = depth_to_points(&depth, 80, 100, &CAMERA, &options);
        assert_eq!(cropped_then_thinned.len(), 50);

        let thinned = thin(
            depth_to_points(&depth, 80, 100, &CAMERA, &CloudOptions::default()),
            50,
        );
        let thinned_then_cropped = thinned
            .iter()
            .filter(|p| {
                let (u, v) = (p[0] * 100.0 / p[2] + 50.0, p[1] * 100.0 / p[2] + 40.0);
                u.round() < 10.0 && v.round() < 10.0
            })
            .count();
        assert!(thinned_then_cropped < 5, "{thinned_then_cropped}");
    }

    #[test]
    fn decimation_keeps_one_pixel_per_block_of_the_crop() {
        let depth = vec![2.0f32; 80 * 100];
        let options = CloudOptions {
            roi: Some([10, 10, 50, 30]),
            decimation: 4,
            ..Default::default()
        };
        assert_eq!(
            depth_to_points(&depth, 80, 100, &CAMERA, &options).len(),
            10 * 5
        );
    }

    #[test]
    fn a_point_seen_past_the_edge_of_a_nearer_one_is_not_an_anchor() {
        // A far point two pixels from a near one: from the camera's viewpoint it is behind.
        let near = [0.0, 0.0, 1.0];
        let far = [0.1, 0.0, 5.0]; // u = 50 + 100 * 0.02 = 52
        let anchors = visible_anchors(&[near, far], &CAMERA, 80, 100);
        assert_eq!(
            anchors,
            vec![Anchor {
                u: 50.0,
                v: 40.0,
                z: 1.0
            }]
        );
        let alone = visible_anchors(&[far], &CAMERA, 80, 100);
        assert_eq!(alone.len(), 1);
    }

    #[test]
    fn scaling_the_camera_keeps_pixel_centres_aligned() {
        let half = CAMERA.scaled(0.5);
        assert_eq!((half.fx, half.cx, half.cy), (50.0, 24.75, 19.75));
    }
}
