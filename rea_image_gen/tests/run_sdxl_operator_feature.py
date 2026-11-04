"""Run one extracted SDXL operator feature on RNGD and save compact evidence."""

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler_engine.execution_auditor import NPUExecutionAuditor
from compiler_engine.graph_lowerings import install_group_norm_decomposition
from compiler_engine.renegade_diffusion_compiler import (
    _NPUBlock, _NPUConv2dSpatialTaps,
)


FEATURES = {
    "group_norm_320": {
        "signature": "group_norm__bf16__n1_c320_h64_w64_g32",
        "input_shape": [1, 320, 64, 64],
    },
    "conv2d_320": {
        "signature": "conv2d__bf16__n1_c320_h64_w64_o320_k3_s1_p1",
        "input_shape": [1, 320, 64, 64],
        "weight_shape": [320, 320, 3, 3],
    },
    "conv2d_downsample_640": {
        "signature": "conv2d__bf16__n1_c640_h32_w32_o640_k3_s2_p1",
        "input_shape": [1, 640, 32, 32],
        "weight_shape": [640, 640, 3, 3],
    },
    "spatial_conv2d_640": {
        "signature": "spatial_tap_conv2d__bf16__n1_c640_h32_w32_o640_k3_s1_p1",
        "input_shape": [1, 640, 32, 32],
        "weight_shape": [640, 640, 3, 3],
    },
    "time_linear_320": {
        "signature": "linear__bf16__n1_i1280_o320",
        "input_shape": [1, 1280],
        "weight_shape": [320, 1280],
    },
    "linear_640_spatial": {
        "signature": "linear__bf16__m1024_i640_o640",
        "input_shape": [1024, 640],
        "weight_shape": [640, 640],
    },
    "linear_640_sequence": {
        "signature": "linear__bf16__b1_s1024_i640_o640",
        "input_shape": [1, 1024, 640],
        "weight_shape": [640, 640],
    },
    "norm_silu_conv_320": {
        "signature": "group_norm32_silu_conv2d__bf16__n1_c320_h64_w64_k3",
        "input_shape": [1, 320, 64, 64],
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("feature", choices=FEATURES)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    spec = FEATURES[args.feature]
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for stale_name in ("compiler_graphs.jsonl", "furiosa_trace.json", "torch_cpu_trace.json",
                       "auditor_events.jsonl", "device_samples.jsonl", "summary.json",
                       "feature_result.json"):
        (output / stale_name).unlink(missing_ok=True)
    os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = str(output / "compiler_graphs.jsonl")
    torch.manual_seed(20261102)
    value = torch.randn(*spec["input_shape"], dtype=torch.bfloat16)

    if args.feature == "group_norm_320":
        module = torch.nn.GroupNorm(32, 320, dtype=torch.bfloat16)
        expected = F.group_norm(value, 32, module.weight.detach().clone(),
                                module.bias.detach().clone(), module.eps)
        install_group_norm_decomposition()
    elif args.feature == "norm_silu_conv_320":
        class NormSiluConv(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.norm = torch.nn.GroupNorm(32, 320, dtype=torch.bfloat16)
                self.conv = torch.nn.Conv2d(320, 320, 3, padding=1, dtype=torch.bfloat16)

            def forward(self, tensor):
                return self.conv(F.silu(self.norm(tensor)))

        module = NormSiluConv()
        expected = F.conv2d(F.silu(F.group_norm(
            value, 32, module.norm.weight.detach().clone(),
            module.norm.bias.detach().clone(), module.norm.eps)),
            module.conv.weight.detach().clone(), module.conv.bias.detach().clone(), padding=1)
        install_group_norm_decomposition()
    elif args.feature in ("conv2d_320", "conv2d_downsample_640", "spatial_conv2d_640"):
        channels_in = 320 if args.feature == "conv2d_320" else 640
        channels_out = channels_in
        stride = 1 if args.feature in ("conv2d_320", "spatial_conv2d_640") else 2
        module = torch.nn.Conv2d(
            channels_in, channels_out, 3, stride=stride, padding=1,
            dtype=torch.bfloat16,
        )
        expected = F.conv2d(
            value, module.weight.detach().clone(), module.bias.detach().clone(),
            stride=stride, padding=1,
        )
    elif args.feature == "time_linear_320":
        module = torch.nn.Linear(1280, 320, dtype=torch.bfloat16)
        expected = F.linear(value, module.weight.detach().clone(), module.bias.detach().clone())
    else:
        module = torch.nn.Linear(640, 640, dtype=torch.bfloat16)
        expected = F.linear(value, module.weight.detach().clone(), module.bias.detach().clone())

    module.to("furiosa:0")
    if args.feature == "spatial_conv2d_640":
        compiled = _NPUConv2dSpatialTaps(module, "sdxl", f"feature.{args.feature}")
    else:
        compiled = _NPUBlock(module, "sdxl", f"feature.{args.feature}")
    auditor = NPUExecutionAuditor(output, f"sdxl-feature-{args.feature}")
    result = {"feature": args.feature, "signature": spec["signature"],
              "input_shape": spec["input_shape"], "dtype": "torch.bfloat16",
              "status": "failed", "npu_task_events": 0}
    try:
        with auditor:
            with auditor.span(f"model_component::feature.{args.feature}"):
                with torch.inference_mode():
                    actual = compiled(value)
        if actual.device.type != "furiosa":
            raise RuntimeError(f"unexpected output device {actual.device}")
        delta = (actual.cpu().float() - expected.float()).abs()
        allowed = 5e-2 + 5e-2 * expected.float().abs()
        passed = bool(torch.all(delta <= allowed))
        summary = json.loads((output / "summary.json").read_text())
        result.update({
            "status": "npu_and_numeric_pass" if passed else "numeric_mismatch",
            "output_device": str(actual.device),
            "npu_task_events": summary["profiler"]["npu_task_events"],
            "graph_confirmed": summary["counts"]["npu_trace_confirmed_graphs"] > 0,
            "mismatch_elements": int((delta > allowed).sum()),
            "max_absolute_error": float(delta.max()),
            "tolerance": {"rtol": 0.05, "atol": 0.05},
        })
        if not passed or not result["graph_confirmed"]:
            raise RuntimeError("NPU execution evidence or numeric tolerance check failed")
    except Exception as exc:
        result["error"] = str(exc).splitlines()[0][:500]
        summary_path = output / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            result["npu_task_events"] = summary["profiler"]["npu_task_events"]
            result["first_failure_point"] = summary.get("first_failure_point")
    result["result_json"] = str(output / "feature_result.json")
    (output / "feature_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    return 0 if result["status"] == "npu_and_numeric_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
