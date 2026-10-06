//! The embedded model on a real recording: tests/data/hotel_hallway.mcap is five frames of a D455 in a
//! hotel hallway a second apart (colour as JPEG, depth registered into the colour camera, both 424x240).
//! Here Depth Anything is already near metric scale (a ~ 0.8-1.3); in other scenes it overshoots by ~2x.
#![cfg(feature = "embedded-model")]

use depth2depth::{Config, Depth2Depth};

struct Frame {
    rgb: Vec<u8>,
    depth_m: Vec<f32>,
    height: usize,
    width: usize,
}

/// Just enough CDR for sensor_msgs CompressedImage and Image.
struct Cdr<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> Cdr<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, offset: 4 }
    }
    fn u32(&mut self) -> u32 {
        self.offset += (4 - (self.offset - 4) % 4) % 4;
        let value =
            u32::from_le_bytes(self.bytes[self.offset..self.offset + 4].try_into().unwrap());
        self.offset += 4;
        value
    }
    fn bytes(&mut self) -> &'a [u8] {
        let length = self.u32() as usize;
        let value = &self.bytes[self.offset..self.offset + length];
        self.offset += length;
        value
    }
    fn skip_header(&mut self) {
        self.u32();
        self.u32();
        self.bytes();
    }
}

fn frames() -> Vec<Frame> {
    let recording = std::fs::read(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/hotel_hallway.mcap"
    ))
    .unwrap();
    let (mut colors, mut depths) = (Vec::new(), Vec::new());
    for message in mcap::MessageStream::new(&recording).unwrap() {
        let message = message.unwrap();
        let mut cdr = Cdr::new(&message.data);
        cdr.skip_header();
        match message.channel.topic.as_str() {
            "/color/image_raw/compressed" => {
                cdr.bytes(); // format
                colors.push(image::load_from_memory(cdr.bytes()).unwrap().to_rgb8());
            }
            "/depth/image_registered" => {
                let (height, width) = (cdr.u32() as usize, cdr.u32() as usize);
                cdr.bytes(); // encoding: 16UC1
                cdr.offset += 1; // is_bigendian
                cdr.u32();
                let millimeters = cdr.bytes();
                let depth_m = millimeters
                    .chunks_exact(2)
                    .map(|mm| u16::from_le_bytes([mm[0], mm[1]]) as f32 / 1000.0)
                    .collect();
                depths.push((depth_m, height, width));
            }
            _ => {}
        }
    }
    assert_eq!(colors.len(), 5);
    colors
        .into_iter()
        .zip(depths)
        .map(|(rgb, (depth_m, height, width))| {
            assert_eq!(
                (rgb.height() as usize, rgb.width() as usize),
                (height, width)
            );
            Frame {
                rgb: rgb.into_raw(),
                depth_m,
                height,
                width,
            }
        })
        .collect()
}

fn median(mut values: Vec<f32>) -> f32 {
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() / 2]
}

#[test]
fn the_hallway_is_filled_and_matches_the_sensor() {
    // The frames are a second apart, too far for the temporal smoothing of the fit.
    let config = Config {
        ema_new_weight: 1.0,
        ..Config::default()
    };
    let (near, far) = (config.near_m, config.far_m);
    let mut model = Depth2Depth::load(config).unwrap();
    // Every frame printed before any is checked, so a failure shows the whole clip.
    let mut results = Vec::new();
    for (index, frame) in frames().iter().enumerate() {
        let fusion = model
            .fuse(&frame.rgb, &frame.depth_m, frame.height, frame.width)
            .unwrap();
        let prediction = model
            .predict(&frame.rgb, frame.height, frame.width)
            .unwrap();
        let trusted: Vec<usize> = (0..frame.depth_m.len())
            .filter(|&i| (near..=far).contains(&frame.depth_m[i]))
            .collect();
        let error = |depth: &[f32]| {
            median(
                trusted
                    .iter()
                    .map(|&i| (depth[i] - frame.depth_m[i]).abs())
                    .collect(),
            )
        };
        let (raw_error, aligned_error) = (error(&prediction), error(&fusion.aligned));
        let holes = frame.depth_m.len() - trusted.len();
        let kept = fusion.kept_raw.iter().filter(|&&kept| kept).count() as f32
            / frame.depth_m.len() as f32;
        println!(
            "frame {index}: holes {:.0}%, a {:.3} b {:.3}, median error vs sensor: Depth Anything {raw_error:.3} m, aligned {aligned_error:.3} m, kept raw {:.0}%",
            100.0 * holes as f32 / frame.depth_m.len() as f32,
            fusion.a,
            fusion.b,
            100.0 * kept
        );
        results.push((fusion, raw_error, aligned_error, kept));
    }
    let mean_aligned_error = results.iter().map(|r| r.2).sum::<f32>() / results.len() as f32;
    for (fusion, raw_error, aligned_error, kept) in results {
        // Fitting the prediction to the sensor's trusted pixels always brings it closer to them.
        assert!((0.5..1.5).contains(&fusion.a), "a = {}", fusion.a);
        assert!(
            aligned_error < raw_error,
            "aligned {aligned_error} m vs Depth Anything alone {raw_error} m"
        );
        assert!(aligned_error < 0.3, "aligned error {aligned_error} m");
        // Every hole is filled, with depth in the model's range.
        assert!(fusion
            .fused
            .iter()
            .all(|&z| z.is_finite() && z > 0.0 && z <= 20.0));
        assert!((0.3..0.8).contains(&kept), "kept raw {kept}");
    }
    assert!(
        mean_aligned_error < 0.2,
        "mean aligned error {mean_aligned_error} m"
    );
}
