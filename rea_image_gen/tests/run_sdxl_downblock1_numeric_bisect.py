"""Bisect SDXL down_blocks.1 numeric drift at the second resnet/attention boundary."""

import json
import os
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

from compiler_engine.execution_auditor import NPUExecutionAuditor
from compiler_engine.graph_lowerings import install_group_norm_decomposition
from compiler_engine.renegade_diffusion_compiler import _segment_down_block_operators


def compare(actual, expected, label):
    got = actual.cpu().float()
    want = expected.float()
    delta = (got - want).abs()
    allowed = 5e-2 + 5e-2 * want.abs()
    return {
        "component": label, "shape": list(got.shape),
        "mismatch_elements": int((delta > allowed).sum()),
        "element_count": got.numel(), "max_absolute_error": float(delta.max()),
        "max_relative_error": float((delta / want.abs().clamp_min(1e-12)).max()),
    }


def main():
    output = Path(os.environ.get(
        "FURIOSA_REPRO_DIR",
        "reports/rngd_porting/runs/sdxl_downblock1_numeric_bisect",
    ))
    output.mkdir(parents=True, exist_ok=True)
    for name in ("compiler_graphs.jsonl", "furiosa_trace.json", "torch_cpu_trace.json",
                 "auditor_events.jsonl", "device_samples.jsonl", "summary.json",
                 "feature_result.json"):
        (output / name).unlink(missing_ok=True)
    os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = str(output / "compiler_graphs.jsonl")
    os.environ["FURIOSA_DYNAMO_RECOMPILE_LIMIT"] = "128"

    torch.manual_seed(20261102)
    unet = UNet2DConditionModel.from_pretrained(
        "stabilityai/sdxl-turbo", subfolder="unet", torch_dtype=torch.bfloat16,
        variant="fp16", token=os.environ.get("HF_TOKEN"), local_files_only=True,
    )
    block = unet.down_blocks[1]
    hidden = torch.randn(1, 320, 32, 32, dtype=torch.bfloat16)
    temb = torch.randn(1, 1280, dtype=torch.bfloat16)
    encoder = torch.randn(1, 77, int(unet.config.cross_attention_dim), dtype=torch.bfloat16)

    # Build exact CPU inputs to the second resnet and second Transformer2D stage.
    with torch.inference_mode():
        after_resnet0 = block.resnets[0](hidden, temb)
        after_attention0 = block.attentions[0](
            after_resnet0, encoder_hidden_states=encoder, return_dict=False,
        )[0]
        expected_resnet1 = block.resnets[1](after_attention0, temb)
        expected_attention1 = block.attentions[1](
            expected_resnet1, encoder_hidden_states=encoder, return_dict=False,
        )[0]

    install_group_norm_decomposition()
    block.to("furiosa:0")
    _segment_down_block_operators(block, "sdxl", 1)
    auditor = NPUExecutionAuditor(output, "sdxl-down-block-1-numeric-bisect")
    with auditor:
        with auditor.span("model_component::unet.down_blocks.1.resnet1"):
            with torch.inference_mode():
                actual_resnet1 = block.resnets[1](
                    after_attention0.to("furiosa:0"), temb.to("furiosa:0"),
                )
        with auditor.span("model_component::unet.down_blocks.1.attention1"):
            with torch.inference_mode():
                actual_attention1 = block.attentions[1](
                    expected_resnet1.to("furiosa:0"),
                    encoder_hidden_states=encoder.to("furiosa:0"), return_dict=False,
                )[0]

    checks = [
        compare(actual_resnet1, expected_resnet1, "resnet1"),
        compare(actual_attention1, expected_attention1, "attention1"),
    ]
    summary = json.loads((output / "summary.json").read_text())
    result = {
        "status": "npu_and_numeric_pass" if all(c["mismatch_elements"] == 0 for c in checks)
        else "numeric_mismatch",
        "checks": checks,
        "confirmed_graphs": summary["counts"]["npu_trace_confirmed_graphs"],
        "unknown_graphs": summary["counts"]["unknown_graphs"],
        "cpu_confirmed_model_graphs": summary["counts"]["cpu_confirmed_model_graphs"],
        "npu_tasks": summary["profiler"]["npu_task_events"],
        "first_failure_point": summary.get("first_failure_point"),
        "tolerance": {"rtol": 0.05, "atol": 0.05},
    }
    (output / "feature_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "npu_and_numeric_pass":
        raise RuntimeError("second resnet/attention still exceeds CPU tolerance")


if __name__ == "__main__":
    main()
