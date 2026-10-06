//! The model built into the library loads and predicts with nothing on disk.
#![cfg(all(feature = "embedded-model", not(feature = "tensorrt")))]

use depth2depth::{Config, Depth2Depth};

#[test]
fn the_embedded_model_predicts_metric_depth() {
    let (height, width) = (120, 160);
    // A horizontal gradient: something with structure for the model to see.
    let rgb: Vec<u8> = (0..height * width)
        .flat_map(|i| {
            let shade = (255 * (i % width) / width) as u8;
            [shade, shade, 128]
        })
        .collect();
    let model = Depth2Depth::load(Config::default().with_quality(0.5)).unwrap();
    let depth = model.predict(&rgb, height, width).unwrap();
    assert_eq!(depth.len(), height * width);
    assert!(depth.iter().all(|d| d.is_finite() && *d > 0.0 && *d <= 20.0));
}
