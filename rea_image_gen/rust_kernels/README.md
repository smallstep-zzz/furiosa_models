# RNGD operator kernel path

This crate starts from the official Furiosa furiosa-opt base-template constant-add kernel and keeps its CPU oracle and NPU test. The first Rust operation is i32[2048] -> i32[2048], adding 1.

src/python_ffi.rs exposes a C ABI for the Python bridge. It deliberately stages host buffers through Furiosa public HostTensor -> HBM -> launch -> HostTensor APIs. This proves Python/PyTorch interoperability but is not zero-copy and should only replace a model op when the transfer cost is justified.

## Build and validate

    cd /root/rea_image_gen/rust_kernels
    cargo test --release --bin constant_add
    cargo furiosa-opt test --release --bin constant_add
    cargo furiosa-opt build --release --lib

`cargo test` is the CPU oracle. `cargo furiosa-opt test` compiles and runs the test on RNGD with cargo-furiosa-opt 0.10.0; `cargo furiosa-opt build` builds the public vISA cdylib. The NPU commands require pinned nightly-2026-05-01, the official SDK driver/runtime, and an available RNGD. The extension ABI reports status codes and stages through host memory; the Python call is not zero-copy.
