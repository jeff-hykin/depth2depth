{
    description = "depth2depth: RGB-guided densification of metric depth images";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        let
            manifest = builtins.fromJSON (builtins.readFile ./model.json);
            # The model files pinned in model.json (those named, or all); the build sandbox has no network, so nix fetches them for build.rs.
            model = pkgs: names: pkgs.linkFarm "depth2depth-model" (map (name: {
                inherit name;
                path = pkgs.fetchurl { url = "${manifest.url}/${name}"; sha256 = manifest.files.${name}; };
            }) names);
            allFiles = builtins.attrNames manifest.files;
        in
        {
            # For a crate2nix (buildRustCrate) build that depends on this crate:
            #   defaultCrateOverrides = pkgs.defaultCrateOverrides // { depth2depth = depth2depth.lib.crateOverride { inherit pkgs; }; };
            # It hands build.rs the model, and with the `tensorrt` feature CUDA + TensorRT (unfree: the caller's pkgs must allow it).
            lib.crateOverride = { pkgs, cudaPackages ? pkgs.cudaPackages_12_6 }: attrs:
                let
                    tensorrt = builtins.elem "tensorrt" (attrs.features or []);
                    # The same choice build.rs makes: TensorRT gets the ONNX and this architecture's prebuilt engine (if any),
                    # candle the safetensors.
                    prebuilt = manifest.prebuilt_engines.${pkgs.stdenv.hostPlatform.parsed.cpu.name} or null;
                    names =
                        if !tensorrt then [ "dinov2_vits14.safetensors" "da2_head_vits.safetensors" ]
                        else [ "da2_metric_hypersim_vits.onnx" ] ++ pkgs.lib.optional (prebuilt != null) prebuilt;
                    # build.rs wants CUDA_HOME/{include,lib64} and TENSORRT_ROOT/{include,lib}; nvcc carries crt/ in CUDA 12.6.
                    cudaHome = pkgs.symlinkJoin {
                        name = "cuda-home";
                        paths = [ cudaPackages.cuda_cudart cudaPackages.cuda_nvcc ];
                        postBuild = "ln -s lib $out/lib64";
                    };
                    tensorrtRoot = pkgs.symlinkJoin { name = "tensorrt-root"; paths = cudaPackages.tensorrt.all; };
                in
                { DEPTH2DEPTH_MODEL_DIR = model pkgs names; }
                // pkgs.lib.optionalAttrs tensorrt { CUDA_HOME = cudaHome; TENSORRT_ROOT = tensorrtRoot; };
        }
        // flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs { inherit system; };
            in
            {
                packages.model = model pkgs allFiles;
                devShells.default = pkgs.mkShell {
                    packages = with pkgs; [
                        rustc
                        cargo
                        clippy
                        rustfmt
                        pkg-config
                        ffmpeg
                    ];
                    DEPTH2DEPTH_MODEL_DIR = model pkgs allFiles;
                    # CUDA/cuDNN intentionally come from the host system
                    # (JetPack on Jetson, the NVIDIA toolkit elsewhere).
                    shellHook = ''
                        export PATH=/usr/local/cuda/bin:$PATH
                    '';
                };
            }
        );
}
