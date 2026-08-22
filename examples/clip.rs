//! Run the fusion pipeline over a recorded clip and write a 3-panel video
//! (rgb | raw depth | fused) via ffmpeg.
//!
//! Usage:
//!   cargo run --release --features cuda --example clip -- \
//!     <dinov2.safetensors> <head.safetensors> <rgb_dir> <depth_mm.npy> <out.mp4> [f16]

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use anyhow::{bail, Context, Result};
use candle::{DType, Device};
use depth2depth::{Config, Depth2Depth};

const FAR_M: f32 = 6.0;

fn load_npy_u16(path: &str) -> Result<(Vec<u16>, Vec<usize>)> {
    let bytes = std::fs::read(path)?;
    if &bytes[..6] != b"\x93NUMPY" {
        bail!("not an npy file");
    }
    let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
    let header = std::str::from_utf8(&bytes[10..10 + header_len])?;
    if !header.contains("'<u2'") {
        bail!("expected u16 npy, header: {header}");
    }
    let shape_str = header
        .split("'shape':")
        .nth(1)
        .context("no shape")?
        .split('(')
        .nth(1)
        .context("no (")?
        .split(')')
        .next()
        .context("no )")?;
    let shape: Vec<usize> = shape_str
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let data = bytes[10 + header_len..]
        .chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect();
    Ok((data, shape))
}

// Polynomial approximation of the turbo colormap (Google AI blog, 2019).
fn turbo(t: f32) -> [u8; 3] {
    let x = t.clamp(0.0, 1.0);
    let r = 0.13572138 + x * (4.61539260 + x * (-42.66032258 + x * (132.13108234 + x * (-152.94239396 + x * 59.28637943))));
    let g = 0.09140261 + x * (2.19418839 + x * (4.84296658 + x * (-14.18503333 + x * (4.27729857 + x * 2.82956604))));
    let b = 0.10667330 + x * (12.64194608 + x * (-60.58204836 + x * (110.36276771 + x * (-89.90310912 + x * 27.34824973))));
    [
        (r.clamp(0.0, 1.0) * 255.0) as u8,
        (g.clamp(0.0, 1.0) * 255.0) as u8,
        (b.clamp(0.0, 1.0) * 255.0) as u8,
    ]
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 6 {
        bail!("usage: clip <dinov2.safetensors> <head.safetensors> <rgb_dir> <depth_mm.npy> <out.mp4> [f16]");
    }
    let dtype = if args.get(6).map(String::as_str) == Some("f16") {
        DType::F16
    } else {
        DType::F32
    };
    let device = Device::cuda_if_available(0)?;
    println!("device {device:?} dtype {dtype:?}");
    let mut d2d = Depth2Depth::new(&args[1], &args[2], device, dtype, Config::default())?;

    let mut rgb_paths: Vec<_> = std::fs::read_dir(&args[3])?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "jpg"))
        .collect();
    rgb_paths.sort();
    let (depth_mm, shape) = load_npy_u16(&args[4])?;
    let (h, w) = (shape[1], shape[2]);
    assert_eq!(shape[0], rgb_paths.len());

    let mut ffmpeg = Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", &format!("{}x{}", 3 * w, h), "-r", "15", "-i", "-",
            "-c:v", "libx264", "-crf", "27", "-pix_fmt", "yuv420p", &args[5],
        ])
        .stdin(Stdio::piped())
        .spawn()?;
    let mut ff_in = ffmpeg.stdin.take().context("ffmpeg stdin")?;

    let mut total_ms = 0f64;
    let mut panel = vec![0u8; 3 * w * h * 3];
    let n_frames = rgb_paths.len();
    for (i, path) in rgb_paths.iter().enumerate() {
        let img = image::open(path)?.to_rgb8();
        let raw: Vec<f32> = depth_mm[i * h * w..(i + 1) * h * w]
            .iter()
            .map(|&mm| mm as f32 / 1000.0)
            .collect();

        let t0 = Instant::now();
        let fusion = d2d.fuse(img.as_raw(), &raw, h, w)?;
        total_ms += t0.elapsed().as_secs_f64() * 1000.0;

        for y in 0..h {
            let row = &mut panel[y * 3 * w * 3..(y + 1) * 3 * w * 3];
            row[..3 * w].copy_from_slice(&img.as_raw()[y * 3 * w..(y + 1) * 3 * w]);
            for x in 0..w {
                let j = y * w + x;
                let raw_px = if raw[j] < 0.3 || raw[j] > FAR_M {
                    [0, 0, 0]
                } else {
                    turbo(raw[j] / FAR_M)
                };
                row[3 * (w + x)..3 * (w + x) + 3].copy_from_slice(&raw_px);
                let fused_px = turbo(fusion.fused[j] / FAR_M);
                row[3 * (2 * w + x)..3 * (2 * w + x) + 3].copy_from_slice(&fused_px);
            }
        }
        ff_in.write_all(&panel)?;
        if i % 50 == 0 {
            println!(
                "{i}/{n_frames}  {:.1} ms/frame  a {:.3} b {:+.3}",
                total_ms / (i + 1) as f64,
                fusion.a,
                fusion.b
            );
        }
    }
    drop(ff_in);
    ffmpeg.wait()?;
    println!(
        "done: {n_frames} frames -> {}  {:.1} ms/frame (incl. warmup)",
        args[5],
        total_ms / n_frames as f64
    );
    Ok(())
}
