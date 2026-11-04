# NPU model porting process

This is the repeatable workflow for bringing a PyTorch image-generation model to Furiosa RNGD. It defines evidence gates; it does not promise that every model or operator is supported. A model is NPU-ready only when every required model graph is accounted for and the requested generation succeeds with CPU model fallback disabled.

## 1. Intake and baseline

If direct SSH cannot resolve, inspect the project's access instructions and authenticated Finder/SFTP mount. A live mount can support remote project file reads and writes, but it does not provide a shell or prove that a test ran. Run compiler, profiler, and NPU commands only through an authenticated remote shell. Do not bypass SSH host-key verification when trying an alternate address.

Record the exact model ID/revision, license, pipeline components, target image size/steps, parameter and activation memory estimates, installed Python/PyTorch/Furiosa/driver/firmware versions, device memory, and existing compiler settings. Load component metadata through the model API; do not scan checkpoint or model binary contents to infer operators. Preserve known-good settings and tests (in this project: PE fusion 8 and `MeshKind.Single()`). Do not upgrade the SDK as part of diagnosis.

First run the existing CPU tests and one known-good NPU control. Save command, exit status, versions, and compact JSON. Use the existing operator compatibility registry and per-operator tests before adding new routes.

## 2. Build a model-operator corpus without reading weights

Use the framework graph boundary: component/module traversal, FX/Dynamo backend graph capture, actual input metadata, and model config. Record per component/graph:

- stable graph ID, component name, original and lowered operator names;
- input shape, dtype, stride, and device from live tensors;
- capture, compile start/return/failure, input preparation, first/repeat execution start/return;
- compiler placement and errors.

Do not infer NPU execution from placement labels or `device=furiosa` alone. Do not force fused runtime kernels into one-to-one FX-op mappings. Group repeated calls by graph and compile instance.

## 3. Establish strict execution evidence

Enable the installed SDK's official NPU trace and CPU profiler; inspect the installed version/API rather than assuming another accelerator generation's helpers. Keep CPU model fallback disabled. Capture CPU ATen events with spans that distinguish explicit CPU model work, CPU host work (tokenizer, scheduler, data preparation), compiler host work, and unknown model-context work.

A graph is **NPU-confirmed** only when a runtime Task event correlates to its graph/component (and compile instance when available). A CPU model operation is confirmed only with an explicit CPU placement/fallback span and CPU operator evidence. A returned call with no trace, active compile with no return, or execution start with no return is **unknown/stalled**, not a pass. SMI process mapping, PE occupancy, memory, utilization, successful compilation, and elapsed time are supporting diagnostics, never execution proof.

Save raw vendor and CPU traces separately. Feed reports and LLMs only compact counts, inputs, phase timings, and the first incomplete graph.

## 4. Walk components once and stop at the first blocker

Run the pipeline in strict NPU-only mode with component/graph auditing and a bounded timeout. Include the first text encoder(s), UNet, and VAE in the required graph set. The first incomplete graph is the sole active failure target. Avoid restarting a whole generation to test guesses: use its captured graph/input metadata to construct a minimal reproducer, and use CPU stack dumps plus phase events when it stalls. If a component can be isolated using the framework/module boundary, exercise that slice with the recorded shapes and dtypes.

Classify the first blocker precisely:

1. memory/model loading;
2. graph capture or unsupported tracing;
3. compiler start with no return / compile error;
4. input preparation or execution-object creation stall;
5. execution submitted but no correlated runtime Task / no return;
6. CPU fallback;
7. numerical mismatch or output postprocessing.

For preparation and execution stalls, inspect stack dumps and SMI/process snapshots; do not rerun the entire model until the minimal graph explains the stall.

## 5. Fix one operator route at a time

For the first failing graph, minimize to the smallest failing op or subgraph while preserving real shapes, dtypes, layouts, and relevant constants. Compare a CPU reference, compiler output, and strict NPU output. Use this order:

1. official installed Furiosa SDK/compiler functionality;
2. public PyTorch decomposition or graph rewrite;
3. already-supported public TCL kernels;
4. a new public TCL kernel;
5. official public Rust/vISA API where available and version-compatible.

Do not jump straight to custom kernels. For every new decomposition/kernel, add a focused unit test with CPU reference, edge shapes/dtypes, NPU trace confirmation, and numeric tolerance. Keep a route only if it compiles, executes on NPU, and passes the comparison. An exact-shape unsupported route should remain an explicit expected failure until fixed; never relabel CPU fallback as NPU success. Do not use REA or inspect/patch proprietary binaries in this workflow.

Use parallel CPU compilation where the server has spare cores (for the current 128-core host, 126 Rayon/OMP threads were used). Do not run competing NPU tests in parallel on the same device: they contaminate utilization and trace attribution.

## 6. Advance through the model and accept only complete runs

After the first blocker is fixed and its focused NPU test passes, continue the same strict audited pipeline to the next graph. Keep the successful path unchanged. Continue until all required text encoder, UNet, and VAE graphs have NPU Task evidence, no model CPU fallback or unknown graph remains, the image is produced, and output sanity checks pass. Tokenization, scheduling, and host data preparation may run on CPU but must remain classified as host work.

If memory cannot hold a one-card model, record actual memory/load failure and drop or re-scope that model; do not treat partial component execution as complete. For an NPU test, report actual per-graph Task counts, CPU model ops, unknown ops, load/compile/preparation/execution timings, shapes/dtypes, PE fusion, memory, output validation, and first blocker.

## 7. Artifacts and resume rule

For this project store summaries in `reports/rngd_porting/`:

- `STATUS.md` starts with verified outcome and first blocker;
- `MODEL_MATRIX.md` records per model/component load, compile, NPU execution, CPU execution, and numeric status;
- `FAILURES.jsonl` records exact graph/input/SDK/command/error plus fact vs hypothesis;
- `REA_BACKLOG.md` stores only unresolved public-API questions;
- a compact operator corpus and per-run `summary.json` contain metadata/counts;
- raw traces remain beside the run and are omitted from review bundles unless requested.

On resume, read STATUS and the latest summary, identify the first unverified graph, and continue there. Do not repeat successful controls or completed components without a reason.

## Current SDXL Turbo checkpoint

The recorded run uses `stabilityai/sdxl-turbo`. Four first-text-encoder FX graphs have official trace evidence (8 Task events total), and the diagnostic intentionally stopped after CLIP layer 0. The full image is not verified. The first recorded whole-pipeline blocker is UNet `conv_in`: strict compiler/cache decoding failed and the exact TCL tactic search failed. Resume by fixing/reproducing that graph, then advance into the rest of UNet and VAE. Current run details are in `reports/rngd_porting/STATUS.md`, `MODEL_MATRIX.md`, `sdxl_operator_corpus.json`, and `runs/sdxl_layer0_audit_final/summary.json`.
