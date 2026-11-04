"""Audit the SDXL VAE decode operator path on RNGD with a CPU reference."""

import json
import os
from pathlib import Path

import torch
from diffusers import AutoencoderKL

from compiler_engine.execution_auditor import NPUExecutionAuditor
from compiler_engine.graph_lowerings import install_group_norm_decomposition
from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes


def main():
    output = Path(os.environ.get(
        "FURIOSA_REPRO_DIR", "reports/rngd_porting/runs/sdxl_vae_decode_feature"
    ))
    output.mkdir(parents=True, exist_ok=True)
    for name in ("compiler_graphs.jsonl", "furiosa_trace.json", "torch_cpu_trace.json",
                 "auditor_events.jsonl", "device_samples.jsonl", "summary.json",
                 "feature_result.json"):
        (output / name).unlink(missing_ok=True)
    os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = str(output / "compiler_graphs.jsonl")
    torch.manual_seed(20261102)
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "126")))
    install_group_norm_decomposition()
    _configure_fused_pes()
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/sdxl-turbo", subfolder="vae", torch_dtype=torch.bfloat16,
        variant="fp16", token=os.environ.get("HF_TOKEN"), local_files_only=True,
    )
    latents = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16)
    with torch.no_grad():
        expected = vae.decode(latents, return_dict=False)[0]
    vae.to("furiosa:0")
    auditor = NPUExecutionAuditor(output, "sdxl-vae-decode-feature")
    with auditor:
        with auditor.span("model_component::vae.decode"):
            with torch.no_grad():
                actual = vae.decode(latents.to("furiosa:0"), return_dict=False)[0]
    summary = json.loads((output / "summary.json").read_text())
    delta = (actual.cpu().float() - expected.float()).abs()
    allowed = 5e-2 + 5e-2 * expected.float().abs()
    scope = summary.get("npu_execution", {}).get("by_scope", {}).get(
        "model_component::vae.decode", {}
    )
    result = {
        "status": "npu_and_numeric_pass" if (
            actual.device.type == "furiosa" and scope.get("operator_task_events", 0) > 0
            and bool(torch.all(delta <= allowed))
        ) else "failed",
        "input_shape": list(latents.shape), "input_dtype": str(latents.dtype),
        "output_shape": list(actual.shape), "output_dtype": str(actual.dtype),
        "output_device": str(actual.device),
        "npu_task_events": summary["profiler"]["npu_task_events"],
        "vae_scope": scope,
        "confirmed_cpu_model_ops": summary["cpu"]["attributed_aten_cpu_ops"]["cpu_model_component"],
        "unknown_graphs": summary["counts"]["unknown_graphs"],
        "max_absolute_error": float(delta.max()),
        "mismatch_elements": int((delta > allowed).sum()),
        "tolerance": {"rtol": 0.05, "atol": 0.05},
    }
    (output / "feature_result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
    if result["status"] != "npu_and_numeric_pass":
        raise RuntimeError("VAE decode lacked NPU Task evidence or exceeded CPU tolerance")


if __name__ == "__main__":
    main()
