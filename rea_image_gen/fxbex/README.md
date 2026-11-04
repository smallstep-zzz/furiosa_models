# Qwen3.5-9B portable FXBEX

- Package: [model.fxbex](https://huggingface.co/softandsweet/redhatai-qwen3-5-9b-fp8-dynamic-rngd-fxbex/resolve/a2d7585a4291ba8d7f6710ae9ec0db09fb5284ce/releases/2026.4.0-portable1/440c02aac9e9ac679a247f9efe9eb3bc8fcc9007204c4ee697210104c54d0774/model.fxbex)
- Model card: <https://huggingface.co/softandsweet/redhatai-qwen3-5-9b-fp8-dynamic-rngd-fxbex>
- Size: 13,544,873,963 bytes (~13.5 GB)
- SHA256: `440c02aac9e9ac679a247f9efe9eb3bc8fcc9007204c4ee697210104c54d0774`
- Source: [RedHatAI/Qwen3.5-9B-FP8-dynamic](https://huggingface.co/RedHatAI/Qwen3.5-9B-FP8-dynamic) at `790f0576d2d77dd5322aa0603a470bd9e3a3d1f6`
- Package: standalone FXBEX with bundled weights, tokenizer/configuration, and Python adapter/runtime
- Runtime: Furiosa RNGD, SDK 2026.4.0, BF16, fusion8, context 512; adapter `qwen35_replay` (runtime 0.4.0)
- License: source weights Apache-2.0; Apache license text is included in the HF release. Bundled adapter has no separate license declaration. Furiosa SDK/driver binaries are not included.

## NPU test

The exact package was tested in a fresh process on Furiosa RNGD with Hugging Face caches empty and offline mode enabled. Filesystem auditing recorded zero reads from the pre-existing HF cache.

- Load: 254.99 s
- Acceptance: cosine 0.999929, RMSE 0.044703, top-1 match
- Runtime: 1,562 NPU tasks; 0 CPU-routed tasks; 0 unknown graphs
- Chat: `What is 2+2?` → `4`; follow-up `multiply by 3` → `12`; repeat prompts returned `4`
- Compiler calls: 0; replay hits: 511

This confirms the tested text path and environment only. It does not establish behavior for other prompts, context lengths, or SDK/driver versions. The server-side full run report is under `/root/rngd-model-forge/reports/rngd_porting/runs/fxbex_portable_offline_20261104/result.json`.
