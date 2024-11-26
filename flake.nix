{
  description = "Flake utils demo";

  inputs.flake-utils.url = "github:numtide/flake-utils";
  inputs.crane.url = "github:ipetkov/crane";
inputs.fenix.url = "github:nix-community/fenix";
inputs.nixpkgs.url = "github:willcohen/nixpkgs/emscripten-3.1.67";

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    crane, fenix,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system; overlays=[
(final: prev: {
binaryen=prev.binaryen.overrideAttrs (_: {doCheck=false;});
})
];};
toolchain = with fenix.packages.${system};
          combine [
            minimal.rustc
            minimal.cargo
            targets.wasm32-unknown-emscripten.latest.rust-std
          ];

        craneLib = (crane.mkLib pkgs).overrideToolchain toolchain;
      in rec {
        packages = rec {
          wheel = craneLib.buildPackage rec {
            src = craneLib.cleanCargoSource ./.;
            cargoArtifacts = craneLib.buildDepsOnly {inherit src nativeBuildInputs CARGO_BUILD_TARGET; PYO3_CROSS_PYTHON_VERSION="3.12";};
          CARGO_BUILD_TARGET = "wasm32-unknown-emscripten";
            buildPhaseCargoCommand = "maturin build --release --target -i python3.12";
            installPhaseCommand = "mv target/wheels $out";
            nativeBuildInputs = [pkgs.python3 pkgs.maturin toolchain pkgs.emscripten];
          };
          python-lib = with pkgs;
            python3Packages.buildPythonPackage {
              pname = "markov";
              version = "0.1.0";
              src = wheel;
              dontUnpack = true;
              buildPhase = ''
                cp -r $src dist
                chown $(whoami) -R dist
                chmod +w -R dist
              '';
            };
          patched-python = pkgs.python3.override {
            packageOverrides = self: super: {
              markov = python-lib;
            };
          };
        };
        devShells.default = with pkgs;
          mkShell {
            buildInputs = [black tev (packages.patched-python.withPackages (ps: [ps.markov ps.ipython ps.numpy ps.pillow ps.scikit-image ps.ffmpeg-python ps.zstandard]))];
          };
devShells.build = with pkgs; mkShell {buildInputs=[python3 maturin emscripten];};
devShells.web = with pkgs; mkShell {buildInputs=[python3 maturin emscripten toolchain];};
      }
    );
}
