# rea_image_gen

A source-only snapshot of `/root/rea_image_gen` plus metadata for its portable Qwen3.5 FXBEX release.

The source snapshot includes model-serving scripts, compiler helpers, tests, Rust kernel sources, and requirements. It omits local credentials, generated images, runtime outputs, multi-gigabyte execution reports, Rust build output, and the nested `rea` checkout.

Model packages are excluded from Git by [`.gitignore`](.gitignore). The standalone FXBEX, including model weights, tokenizer/configuration, and bundled Python adapter/runtime, is hosted on [Hugging Face](https://huggingface.co/softandsweet/redhatai-qwen3-5-9b-fp8-dynamic-rngd-fxbex). See [`fxbex/README.md`](fxbex/README.md) for the immutable artifact URL, SHA256, test evidence, and license notes.

This is an FXBEX package, not an official `.fxb` artifact. The verified scope is text generation on Furiosa RNGD with SDK 2026.4.0, BF16 execution, fusion8, context 512, and the pinned source revision documented in the release metadata.
