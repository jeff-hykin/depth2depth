//! Build (or load) the TensorRT engine for one model input size, e.g. to prebuild it for model.json:
//!   cargo run --release --features tensorrt --example engine -- 448 560
//! It lands in `depth2depth::engine_cache_dir()`.
fn main() {
    let args: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("height width"))
        .collect();
    let [height, width] = args[..] else {
        panic!("usage: engine <height> <width> (multiples of 14)");
    };
    let started = std::time::Instant::now();
    depth2depth::Depth2Depth::load(depth2depth::Config {
        model_h: height,
        model_w: width,
        ..Default::default()
    })
    .unwrap();
    println!(
        "engine for {height}x{width} ready in {:.0?}, in {}",
        started.elapsed(),
        depth2depth::engine_cache_dir().display()
    );
}
