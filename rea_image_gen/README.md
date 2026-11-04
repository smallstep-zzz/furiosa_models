# rea_image_gen

A source-only snapshot of `/root/rea_image_gen` plus its existing Qwen3.5-9B FXBEX package under `FXB/`.

The snapshot includes the model-serving scripts, compiler helpers, tests, Rust kernel sources, and requirements. It omits local credentials, generated images, runtime outputs, multi-gigabyte execution reports, Rust build output, and the nested `rea` checkout.

The `.fxbex` file is an FXBEX package, not an official Furiosa `.fxb` artifact. It is a lightweight package and does not include model weights. Loading requires the matching `qwen35_replay` adapter, compatible Hugging Face checkpoint, and Furiosa SDK 2026.4.0; see the package README for model revision, dtype, fusion, and context limit.

The source server's `fxbex validate` passed integrity and static readiness checks. That command reports inference as not executed; this publication does not claim a fresh NPU inference acceptance. `FXB/*export-status.json` records the separate official FXB export result (`EXPORT_BRIDGE_GAP`).
