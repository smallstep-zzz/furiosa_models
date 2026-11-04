---
name: rngd-model-porting
description: Port or audit PyTorch image-generation models on Furiosa RNGD, with strict NPU-only execution evidence and graph-by-graph operator support work.
---

# Furiosa RNGD model porting

Use this skill to make progress on a real model port and to determine whether its required model computations actually ran on RNGD. Never promise universal operator support: report what the installed SDK and device prove for the exact model, shapes, and dtypes.

## Required behavior

- Inspect the project's contract, existing implementation, tests, latest status, and installed SDK versions before changing code. Preserve known-good compiler settings and routes.
- If SSH fails, inspect documented authenticated Finder/SFTP mounts for project file access. A file mount is not a remote shell and is not execution evidence; run NPU/compiler commands only through a verified remote shell. Never bypass host-key verification to reach an alternate IP.
- Discover operators through framework components and captured FX/Dynamo graphs plus live input metadata. Never scan checkpoint/weight binaries to discover operators.
- Disable CPU model fallback for NPU acceptance runs. Allow CPU host work only when explicitly classified (tokenization, scheduler control, data preparation, postprocessing).
- Confirm NPU execution only from the installed official runtime trace correlated to graph/component and compile instance when possible. Compilation, device placement, SMI utilization/process mapping, or a returned Python call alone is insufficient. Leave missing evidence `unknown`.
- Follow the installed SDK's supported path first, then public PyTorch decomposition, existing public TCL, new public TCL, and compatible official Rust/vISA. Test each operator route against a CPU reference and require NPU trace evidence. No proprietary binary analysis or REA.
- Find the first incomplete graph, capture a minimal reproduction at its real shape/dtype, and fix one blocker at a time. Use stack snapshots and phase timings to separate compile, preparation, execution, and CPU fallback. Do not repeatedly run full generations to test unverified hypotheses.
- Parallelize independent CPU reference/export/compile jobs with a global CPU and RAM budget. Permit concurrent NPU correctness jobs only after a small two-process experiment proves SDK-supported resource sharing, isolated traces, and sufficient memory. Use one exclusive job for performance measurements and fusion=8 integration; never assume a card can be partitioned.
- Check all explicit `.cpu()` / `.to("cpu")` calls inside model forwards. Padding, unfolding, window packing and tensor arithmetic are model computation even when labelled host. Graph coverage alone does not establish NPU-only support.
- Complete one strict full pipeline before calling a model supported: all required text encoder, denoiser/UNet, and decoder/VAE graphs accounted for, no unknown/model-CPU graph, image produced, and output sanity checked.
- Keep raw profiler traces separate; summarize per-graph verdicts, counts, shape/dtype, timings, memory, device configuration, and first failure in small JSON and project reports.

## Workflow details

Read [references/porting-checklist.md](references/porting-checklist.md) for the detailed gates, diagnosis sequence, report fields, and the active SDXL Turbo checkpoint. Use repository-specific runbooks and the installed SDK's public API for actual commands.

## Operator-first development and performance gates

Read [references/operator-workflow.md](references/operator-workflow.md) when adding a model, choosing TCL/vISA, diagnosing low utilization, or scheduling parallel feature tests.
