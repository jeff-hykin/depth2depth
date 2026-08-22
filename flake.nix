{
    description = "depth2depth: RGB-guided densification of metric depth images";

    inputs = {
        nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";
    };

    outputs = { self, nixpkgs, flake-utils }:
        flake-utils.lib.eachDefaultSystem (system:
            let
                pkgs = import nixpkgs { inherit system; };
            in
            {
                devShells.default = pkgs.mkShell {
                    packages = with pkgs; [
                        rustc
                        cargo
                        clippy
                        rustfmt
                        pkg-config
                        ffmpeg
                    ];
                    # CUDA/cuDNN intentionally come from the host system
                    # (JetPack on Jetson, the NVIDIA toolkit elsewhere).
                    shellHook = ''
                        export PATH=/usr/local/cuda/bin:$PATH
                    '';
                };
            }
        );
}
