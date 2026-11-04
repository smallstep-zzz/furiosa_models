# Diffusers graph compiler adapter

## Workflow

This package adds an FX graph adaptation layer in front of the installed Furiosa PyTorch compiler. It does not scan checkpoint/model binary bytes. Operator inventories come from `torch.compile` FX graphs created while the cached Diffusers pipeline executes. `FURIOSA_GRAPH_AUDIT_PATH=/path/to/audit.jsonl` saves the pre/post-lowering operator names, component, graph hash, placement, and first compiler error. The audit file contains graph metadata only, not tensor values or weights.

The adapter sends captured graphs to Furiosa torch 2026.4 with CompilerConfig and the vision tactic. It configures eight fused PEs with MeshKind.Single before moving components to furiosa:0. On strict compile failure it tries per-op NPU execution, then CPU fallback; explicit Furiosa device constants are rewritten for CPU fallback and outputs return to furiosa:0.

CPU compile parallelism is configured with `RAYON_NUM_THREADS` and `OMP_NUM_THREADS`. The current server has 128 CPU cores; the run scripts reserve two and set both to 126.

## Required compiler-debug workflow

Follow these steps in order whenever a model graph fails to compile. Do not start REA analysis just because one compile attempt failed.

1. **Record the failure precisely.** Save the SDK/compiler version, compiler config, graph hash, component, operator overloads, shapes, dtypes, and device placement from the captured FX graph. Do not inspect checkpoint bytes to discover operators.
2. **Check public SDK support.** Read the installed SDK API/type stubs, official documentation, release notes, and shipped examples for the operator and device/layout behavior. Confirm whether the installed release has a supported compiler, runtime, or kernel API.
3. **Build a minimal reproducer.** Reduce the failing model graph to the smallest graph that retains the same operator, shape/dtype, and device mismatch. Put the reproducer in a focused test so later SDK upgrades can rerun it quickly.
4. **Decompose unsupported fused operators.** Rewrite the operator into the smallest semantically equivalent set of primitives already accepted by the backend. Add CPU reference tests for semantics and an opt-in RNGD test that checks output device and numerical tolerance.
5. **Implement a public kernel when decomposition is insufficient.** Prefer documented Furiosa interfaces such as TCL/furiosa-kernels or another public SDK extension point. Keep model-specific hints and tensor/PE layout explicit. Test the kernel in isolation before adding it to the pipeline.
6. **Retry strict compilation, then runtime.** Record separately whether a graph was compiled as one NPU program, executed through per-op NPU fallback, or placed on CPU. Never count fallback execution as whole-graph compilation.
7. **Use REA only for an unresolved opaque vendor behavior.** Reach this step only after public APIs/docs/examples and a minimal reproducer fail to explain the behavior. Use REA to inspect the relevant compiler/runtime component or binary path, not model checkpoint binaries; preserve concrete evidence and limitations from that analysis. If source or docs already explain the issue, fix the code or report the SDK limitation without REA.

This order keeps investigations small and repeatable, and makes each added lowering/kernel independently testable before a full diffusion pipeline run.

## Compiler changes in this adapter

| Captured operation | Adapter change | Unit coverage | NPU compiler result |
| --- | --- | --- | --- |
| torch.nn.functional.scaled_dot_product_attention | FX rewrite to manual_scaled_dot_product_attention with matmul, transpose, scale, boolean/additive/causal mask, softmax, optional dropout, output matmul, and grouped-query head expansion. | tests/test_graph_lowerings.py: seven semantic/lowering cases. tests/test_npu_backend.py: opt-in NPU execution and CPU numerical comparison. | On SDK 2026.4, plain SDPA runs as npu_eager; the CPU-created-mask graph compiles as npu_compiled. A patternize_llm.rs:892 panic was observed on 2026.2 and is historical; it was not reproduced in the 2026.4 test run.
| Explicit Furiosa device constants on CPU fallback graphs | Recursive FX argument rewrite to CPU before fallback execution. Outputs are copied back to furiosa:0 at the compiled-segment boundary. | test_cpu_fallback_rewrites_explicit_furiosa_device. | Boundary behavior tested on CPU; whole-pipeline fallback still requires integration validation.

`Qwen3-VL` rotary graphs can receive CPU-created cosine/sine factors alongside RNGD query/key tensors. The adapter moves floating CPU graph inputs to RNGD before strict/per-op execution, and the actual Qwen rotary function passes a focused mixed-device per-op NPU test. The one-step Qwen pipeline still fails earlier in strict FakeTensor propagation; its bounded run recorded only the 4-node embedding graph as npu_compiled, emitted CPU fallback/type-conversion warnings, showed 0% PE utilization, and produced no image.

All other operators currently pass through unchanged to the installed Furiosa backend. The inventory is evidence about captured graphs, not a promise that every listed operator has an NPU lowering.

## Model graph evidence captured so far

The inspected `text_encoder` FX graphs from cached SDXL-Turbo and FLUX.1-dev contained these high-level operations:

- Function calls: `add`, `getitem`, `le`, `linear`, `mul`, `arange`, `sigmoid`, `embedding`, `layer_norm`, and fused SDPA.
- Methods: `argmax`, `contiguous`, `expand`, `reshape`, `to`, `transpose`, and `view`.

A captured SDXL `text_encoder_2` graph had 1,243 FX nodes and 32 SDPA call sites; the captured SDXL first text encoder graph had 507 nodes and 12 SDPA call sites. Graph hashes and exact pre/post-lowering names are in the JSONL audit output on the server. These are runtime graph observations from the requested model components; model weights were not inspected as raw bytes.

## Current integration status and limitations

- Memory diagnosis: the earlier SDXL UNet failure occurred during `module.to(rngd:0)`, before compiler invocation. The allocator pool was 6,375,342,080 bytes (5.9375 GiB, exactly one eighth of the card's 47.5 GiB), and rejected a 52,428,800-byte allocation after 6,333,647,360 bytes were allocated. That was the default one-PE fusion placement limit, not whole-card OOM and not compiler memory planning. Calling `set_fusion(8)` with `MeshKind.Single()` before importing/using the Furiosa backend fixed placement: a fresh SDXL load-only run placed UNet, VAE, and both text encoders on RNGD without allocator errors. The adapter now configures fusion at import time, before backend initialization. `FURIOSA_FUSION_PES` can set a lower PE count if needed. The earlier allocator measurements and successful load-only run were on SDK 2026.2; on SDK 2026.4 the bounded SDXL attempt did not reach a graph audit or allocate NPU DRAM.
- SDXL-Turbo: a 2026.4 one-step run timed out at 90 seconds before graph audit or image generation. No NPU DRAM allocation or PE activity was observed; this is not evidence of whole-card OOM.
- FLUX.1-dev: the 2026.4 full generation path has not been re-run. Qwen-Image-2.1: 2026.4 loads components and compiles small text-encoder graphs, but a 3,443-node text-encoder graph falls to CPU after a lifted-tensor attribute error; a later mask/index device mismatch aborts one-step generation before an image is produced.
- FLUX 3 Klein: not executable offline because the server cache is incomplete (`model_index.json` is absent); no graph was captured.
- SDPA lowering: seven CPU semantic/lowering cases and five opt-in NPU tests pass on 2026.4. CPU-created-mask and rotary graphs were npu_compiled; plain SDPA used npu_eager. All tested outputs matched CPU PyTorch within rtol=2e-2 and atol=2e-2.

The inter-chip TpMemoryLayout conclusion was established on the older 2026.2 API. The 2026.4 adapter no longer uses that removed layout API; PE fusion controls placement for this single-card setup. Full model output coverage is not complete.

## Image-generation dependency set

requirements-furiosa-image-gen.txt pins the direct versions used by the Diffusers image-generation path. It intentionally excludes Furiosa-LLM and the retired furiosa-models package. The installed target set satisfies the direct pins: PyTorch 2.10.0+cpu satisfies Furiosa torch's torch==2.10.0 requirement, NumPy 2.2.6 satisfies its >=2.2.6 requirement, and Hugging Face Hub 1.33.0 satisfies both Diffusers and Transformers constraints. Gradio 6.31.0 accepts FastAPI 0.162.0.

The remaining global pip check findings belong to packages outside this image-generation set: PyGObject/pycairo, OpenTelemetry, and Furiosa-LLM/FastAPI. They are not introduced by this project's requirements file. Keep the image-generation environment separate from the server's Furiosa-LLM service dependencies.

## Applied Furiosa SDK 2026.4 stack

| Component | Applied version |
| --- | --- |
| furiosa-torch | 2026.4.0 |
| furiosa-kernels, furiosa-kernels-contrib, furiosa-tcc | 2026.4.0 |
| PyTorch / NumPy | 2.10.0+cpu / 2.2.6 |
| RNGD driver / firmware | 2026.4.0 / 2026.4.0 (already matched) |

The incompatible unused furiosa-models 2026.2.0 and furiosa-torch-ext 2026.3.0 packages were removed. The 2026.4 wheel provides TclModule; the official furiosa.tcl.examples.attention example passes on the NPU and matches its PyTorch reference. Five adapter NPU integration tests pass on this stack. Three audited graphs were npu_compiled; plain SDPA used npu_eager. The adapter still uses the tested FX SDPA decomposition; direct substitution with the official TCL attention module remains a separate integration change.

pip check still reports unrelated pre-existing environment conflicts: PyGObject without pycairo, OpenTelemetry API version mismatches, and furiosa-llm versus FastAPI. Those packages were not changed.

## REA findings (pre-upgrade compiler binary)

After public SDK documentation, installed stubs/examples, a focused reproducer, and per-op decomposition had been checked, REA was used on the old compiler-side ELF only, not model files. inspect-rust-binary on furiosa/native_torch.cpython-310-x86_64-linux-gnu.so recorded SHA-256 7449350b1a5d4f42b04a2e1416a26954a5bd9630bf78a6edc82af2398e873fda, but found no .dep-v0 dependency metadata and does not recover functions or runtime behavior. inspect-native-exports found 65 public ELF symbols, including runtime/device calls, but no internal compiler source. Ghidra, Hopper, IDA, and Kuna providers were unavailable. That analysis did not explain the 2026.2 panic and does not describe the upgraded 2026.4 binary.

## Applying this adapter to a newer Furiosa release

Furiosa 2026.3 introduced the TCL kernel framework and furiosa-kernels; 2026.4 is now applied on this server. For the next SDK upgrade, verify the published package set and driver/firmware requirements, rerun the official TCL attention example first, then rerun focused NPU tests and captured model graphs.


## Public TCL and Rust vISA extension path (verified 2026-11-02)

The default path remains the installed Furiosa Torch compiler. For an explicit unsupported-op adapter, use the shared registry in this order: Python decomposition, existing/new public TCL, then Rust vISA. Each implementation must have a CPU oracle, focused shape/type predicate, unit test, and real RNGD numerical test before registration. Route errors must remain visible; do not silently label eager or CPU fallback as compiled NPU execution.

The opt-in registry is compiler_engine/operator_compat.py. resolve_operator loads registrations lazily and selects the first supported route. aten.cat promotion and SDPA FX replacement use this registry. compiler_engine/tcl_kernels.py wraps the installed public convolution example only for exact BF16 specialization [2,3,8,8], filter [5,3,3,3], bias [5], padding 1 and stride 2. This is a verified example adapter, not general conv2d.

The rust_kernels crate uses public furiosa-opt APIs, pinned to furiosa-opt-std 0.10.0 and nightly-2026-05-01. cargo test --release --bin constant_add is CPU oracle; cargo furiosa-opt test --release --bin constant_add runs on physical RNGD; cargo furiosa-opt build --release --lib builds the cdylib. The proof kernel is i32[2048] add-one. Python calls its public C ABI via host staging. It is verified only in a separate process: when Furiosa Torch holds the RNGD, furiosa-opt Device::new returns Device(Reserved(2)). This is not same-process or zero-copy.

Official furiosa-opt tooling requires documented x86_64 Ubuntu, matching pinned Rust nightly, Furiosa SDK and physical NPU; this project uses installed SDK 2026.4.0 without changing it. furiosa-opt is experimental. Recheck official tooling/API and driver requirements per SDK release.

SDXL 2026.4 behavior is in reports/rngd_porting/STATUS.md and FAILURES.jsonl. Exact conv2d overload/shape remains to be captured. Continue public failure workflow; no REA or proprietary-binary analysis is performed in this extension step.
