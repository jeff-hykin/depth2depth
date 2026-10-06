// With `embedded-model` (default), put the model files named in model.json into OUT_DIR for include_bytes!:
// copied from DEPTH2DEPTH_MODEL_DIR when it is set (nix: the build sandbox has no network), else downloaded.
// Either way each file must match model.json's sha256, so the model is pinned with the crate.
//
// With `tensorrt`, compile the C++ shim over TensorRT and link it, CUDA and the ONNX parser.
// CUDA_HOME (default /usr/local/cuda) and TENSORRT_ROOT (default: the system paths JetPack uses) locate them.

use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=src/tensorrt.cpp");
    if std::env::var_os("CARGO_FEATURE_EMBEDDED_MODEL").is_some() {
        fetch_model();
    }
    if std::env::var_os("CARGO_FEATURE_TENSORRT").is_some() {
        build_tensorrt();
    }
}

fn fetch_model() {
    println!("cargo:rerun-if-changed=model.json");
    println!("cargo:rerun-if-env-changed=DEPTH2DEPTH_MODEL_DIR");
    let manifest_path = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("model.json");
    let manifest: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&manifest_path).expect("model.json"))
            .expect("model.json is not JSON");
    let url = manifest["url"].as_str().expect("model.json: url");
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    // TensorRT builds its engine from the ONNX; candle loads the safetensors.
    let names: &[&str] = if std::env::var_os("CARGO_FEATURE_TENSORRT").is_some() {
        &["da2_metric_hypersim_vits_364x448.onnx"]
    } else {
        &["dinov2_vits14.safetensors", "da2_head_vits.safetensors"]
    };
    for name in names {
        let sha256 = manifest["files"][name].as_str().expect("model.json: files");
        println!("cargo:rustc-env=D2D_SHA256_{}={sha256}", name.replace(['.', '-'], "_"));
        let destination = out_dir.join(name);
        // docs.rs builds without network; the docs don't need the bytes.
        if std::env::var_os("DOCS_RS").is_some() {
            std::fs::write(&destination, b"").unwrap();
            continue;
        }
        if destination.exists() && sha256_of(&destination) == sha256 {
            continue;
        }
        match std::env::var_os("DEPTH2DEPTH_MODEL_DIR") {
            Some(dir) => {
                std::fs::copy(Path::new(&dir).join(name), &destination)
                    .unwrap_or_else(|e| panic!("copying {name} from DEPTH2DEPTH_MODEL_DIR: {e}"));
            }
            None => download(&format!("{url}/{name}"), &destination),
        }
        let actual = sha256_of(&destination);
        if actual != sha256 {
            let _ = std::fs::remove_file(&destination);
            panic!("{name}: sha256 {actual}, model.json pins {sha256}");
        }
    }
}

// curl rather than an HTTP crate: no TLS stack to compile, and nix never gets here.
fn download(url: &str, destination: &Path) {
    let partial = destination.with_extension("part");
    let status = std::process::Command::new("curl")
        .args(["--fail", "--location", "--silent", "--show-error", "--retry", "3", "--output"])
        .arg(&partial)
        .arg(url)
        .status()
        .unwrap_or_else(|e| panic!("running curl to download {url}: {e} (or set DEPTH2DEPTH_MODEL_DIR)"));
    if !status.success() {
        panic!("downloading {url} failed ({status}); set DEPTH2DEPTH_MODEL_DIR to a directory holding the files in model.json");
    }
    std::fs::rename(&partial, destination).unwrap();
}

fn sha256_of(path: &Path) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    std::io::copy(&mut std::fs::File::open(path).unwrap(), &mut hasher).unwrap();
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn build_tensorrt() {
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda".into());
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .file("src/tensorrt.cpp")
        .include(format!("{cuda}/include"))
        .warnings(false);
    if let Ok(root) = std::env::var("TENSORRT_ROOT") {
        build.include(format!("{root}/include"));
        println!("cargo:rustc-link-search=native={root}/lib");
    }
    build.compile("d2d_tensorrt");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    for library in ["nvinfer", "nvonnxparser", "cudart"] {
        println!("cargo:rustc-link-lib={library}");
    }
    println!("cargo:rustc-link-lib=stdc++");
}
