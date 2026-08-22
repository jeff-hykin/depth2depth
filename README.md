# depth2depth

Turn a noisy, hole-riddled metric depth image (RealSense-style stereo depth) into a dense one, using the matching RGB frame.

How: [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) (metric, vit-small) predicts dense depth from RGB. That prediction has the right *shape* but unreliable *scale*, so every frame it is affine-fitted (`raw ≈ a·pred + b`, robust two-round least squares) to the trusted pixels of the raw sensor depth. The output keeps the raw sensor value wherever it agrees with the aligned prediction and uses the aligned prediction everywhere else — holes, dropouts, and outliers get filled, real sensor geometry survives untouched. The `(a, b)` fit is smoothed over time (EMA) so the fill doesn't flicker.

Pure Rust, no Python and no ONNX runtime at inference time. Inference runs on [candle](https://github.com/huggingface/candle), so the GPU backend is a cargo feature: `cuda` / `cudnn` (NVIDIA, incl. Jetson), `metal` (Apple), or nothing for CPU.

## Usage

```rust
use depth2depth::{Config, Depth2Depth};
use candle_core::{Device, DType};

let mut d2d = Depth2Depth::new(
    "dinov2_vits14.safetensors",
    "da2_head_vits.safetensors",
    Device::cuda_if_available(0)?,
    DType::F16,
    Config::default(),
)?;

// rgb: HxWx3 u8, raw_depth_m: HxW f32 meters (0 / out-of-range = hole)
let fusion = d2d.fuse(&rgb, &raw_depth_m, height, width)?;
// fusion.fused: dense HxW f32 meters
// fusion.kept_raw: which pixels are untouched sensor readings
// fusion.a, fusion.b: the current affine fit
```

`Config` controls the model input resolution (default 280×504, must be multiples of 14 — smaller is faster), the raw-depth trust range (default 0.3–6 m), the agreement tolerance `max(0.3 m, 10%·z)`, and the EMA weight. `Config::default().with_quality(0.5)` scales the model resolution as a single quality/speed knob. Call `reset()` on scene cuts.

## Model weights

The crate loads two safetensors files converted from the official Depth Anything V2 metric checkpoint (Hypersim indoor, vit-small, `max_depth = 20`):

```sh
# needs: pip install torch safetensors
python tools/convert_weights.py depth_anything_v2_metric_hypersim_vits.pth weights/
```

Checkpoint download: see the [Depth-Anything-V2 metric_depth page](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth). The conversion also synthesizes the (unused) ImageNet classifier head that candle's dinov2 module insists on loading.

## Example

3-panel video (rgb | raw | fused) over a recorded clip:

```sh
cargo run --release --features cuda --example clip -- \
    weights/dinov2_vits14.safetensors weights/da2_head_vits.safetensors \
    rgb_dir/ depth_mm.npy out.mp4 f16
```

`depth_mm.npy` is a `(frames, H, W)` uint16 array of millimeters; `rgb_dir/` holds one jpg per frame. Needs `ffmpeg` on PATH.

## Building

- NVIDIA: `cargo build --release --features cuda` (add `cudnn` if libcudnn is installed — much faster convolutions; on Jetson set `CUDA_COMPUTE_CAP` to your arch, e.g. `87` for Orin).
- Apple: `--features metal`.
- CPU: no features (slow; fine for tests).

`nix develop` gives a shell with the Rust toolchain (CUDA comes from the system, not nix).

## Notes / limitations

- vit-small only, for now. The DPT head and dinov2 modules are vendored from [candle-transformers](https://github.com/huggingface/candle) (Apache-2.0/MIT) with three changes: metric-depth output head (sigmoid × max_depth instead of relu), support for non-square inputs, and a fixed h/w ordering in the positional-embedding interpolation.
- candle's `interpolate2d`/`upsample_nearest2d` are nearest-neighbor where the PyTorch original uses bilinear/bicubic; the per-frame affine refit absorbs most of the difference.
- The affine anchor is not optional: on real indoor scenes the raw DA2 metric prediction can be off by ~2×.

## License

MIT or Apache-2.0, at your option. Vendored candle code is likewise Apache-2.0/MIT, © the candle authors.
