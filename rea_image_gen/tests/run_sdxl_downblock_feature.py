"""Run one extracted SDXL UNet down-block feature on RNGD and compare to CPU."""

import json
import os
from copy import deepcopy
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

from compiler_engine.execution_auditor import NPUExecutionAuditor
from compiler_engine.graph_lowerings import (
    install_group_norm_decomposition,
)
from compiler_engine.renegade_diffusion_compiler import _segment_down_block_operators


def main():
    block_index = int(os.environ.get("FURIOSA_REPRO_BLOCK", "1"))
    output = Path(os.environ.get(
        "FURIOSA_REPRO_DIR",
        f"reports/rngd_porting/runs/sdxl_downblock{block_index}_segmented_feature",
    ))
    output.mkdir(parents=True, exist_ok=True)
    for name in ("compiler_graphs.jsonl", "furiosa_trace.json", "torch_cpu_trace.json",
                 "auditor_events.jsonl", "device_samples.jsonl", "summary.json",
                 "feature_result.json"):
        (output / name).unlink(missing_ok=True)
    os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = str(output / "compiler_graphs.jsonl")

    torch.manual_seed(20261102)
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/sdxl-turbo", subfolder="unet", torch_dtype=torch.bfloat16,
        variant="fp16", token=os.environ.get("HF_TOKEN"), local_files_only=True,
    )
    block = unet.down_blocks[block_index]
    reference = deepcopy(block)
    channels = int(block.resnets[0].norm1.num_channels) if getattr(block, "resnets", None) else int(getattr(block, "in_channels", 320))
    spatial = 64 // (2 ** block_index)
    cross_dim = int(unet.config.cross_attention_dim)
    hidden = torch.randn(1, channels, spatial, spatial, dtype=torch.bfloat16)
    temb = torch.randn(1, int(unet.config.block_out_channels[0]) * 4, dtype=torch.bfloat16)
    encoder = torch.randn(1, 77, cross_dim, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = reference(hidden, temb, encoder_hidden_states=encoder)

    install_group_norm_decomposition()
    block.to("furiosa:0")
    wrapped = _segment_down_block_operators(block, "sdxl", block_index)
    if wrapped == 0:
        raise RuntimeError(f"down_blocks.{block_index} had no segmentable child modules")

    auditor = NPUExecutionAuditor(output, f"sdxl-down-block-{block_index}")
    with auditor:
        with auditor.span(f"model_component::unet.down_blocks.{block_index}"):
            with torch.inference_mode():
                actual = block(
                    hidden.to("furiosa:0"), temb.to("furiosa:0"),
                    encoder_hidden_states=encoder.to("furiosa:0"),
                )
    actual_leaves, actual_spec = torch.utils._pytree.tree_flatten(actual)
    expected_leaves, expected_spec = torch.utils._pytree.tree_flatten(expected)
    if actual_spec != expected_spec or len(actual_leaves) != len(expected_leaves):
        raise RuntimeError("CPU and NPU down-block output structures differ")
    output_checks = []
    for index, (got, want) in enumerate(zip(actual_leaves, expected_leaves)):
        got_f = got.cpu().float()
        want_f = want.float()
        delta = (got_f - want_f).abs()
        allowed = 5e-2 + 5e-2 * want_f.abs()
        output_checks.append({
            "leaf_index": index, "shape": list(got.shape),
            "mismatch_elements": int((delta > allowed).sum()),
            "element_count": got.numel(),
            "max_absolute_error": float(delta.max()),
            "max_relative_error": float((delta / want_f.abs().clamp_min(1e-12)).max()),
        })
    summary = json.loads((output / "summary.json").read_text())
    if summary["counts"]["npu_trace_confirmed_graphs"] < 1:
        raise RuntimeError("No graph-correlated NPU Task event was recorded")
    result = {
        "feature_id": f"sdxl_down_block_{block_index}_segmented",
        "status": (
            "npu_and_numeric_pass"
            if all(item["mismatch_elements"] == 0 for item in output_checks)
            else "numeric_mismatch"
        ),
        "input_shape": list(hidden.shape), "temb_shape": list(temb.shape),
        "encoder_shape": list(encoder.shape), "dtype": str(hidden.dtype),
        "segmented_children": wrapped,
        "npu_tasks": summary["profiler"]["npu_task_events"],
        "confirmed_graphs": summary["counts"]["npu_trace_confirmed_graphs"],
        "unknown_graphs": summary["counts"].get("unknown_graphs", 0),
        "cpu_confirmed_model_graphs": summary["counts"].get("cpu_confirmed_model_graphs", 0),
        "tolerance": {"rtol": 0.05, "atol": 0.05},
        "output_checks": output_checks,
    }
    (output / "feature_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "npu_and_numeric_pass":
        raise RuntimeError("one or more down-block output leaves exceeded tolerance")


if __name__ == "__main__":
    main()
