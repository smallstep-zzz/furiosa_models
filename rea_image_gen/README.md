# rea_image_gen model delivery

This directory contains the existing Qwen3.5-9B FXBEX package from `/root/rea_image_gen/FXB`.

The `.fxbex` file is an FXBEX package, not an official Furiosa `.fxb` artifact. It is a lightweight package and does not include model weights. Loading requires the matching `qwen35_replay` adapter, compatible Hugging Face checkpoint, and Furiosa SDK 2026.4.0; see the package README for the model revision, dtype, fusion, and context limit.

The source server's `fxbex validate` passed integrity and static readiness checks. That command reports inference as not executed; this publication does not claim a fresh NPU inference acceptance. `FXB/*export-status.json` records the separate official FXB export result (`EXPORT_BRIDGE_GAP`).
