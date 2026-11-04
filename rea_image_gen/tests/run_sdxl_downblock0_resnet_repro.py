"""Run SDXL down_blocks.0.resnets.0 in strict NPU mode, without a full denoising pass."""

import json
import os
from copy import deepcopy
from pathlib import Path

import torch
from diffusers import UNet2DConditionModel

from compiler_engine.execution_auditor import NPUExecutionAuditor
from compiler_engine.renegade_diffusion_compiler import _NPUBlock, _segment_down_block_operators
from compiler_engine.graph_lowerings import GroupNormMeanSquare, install_group_norm_decomposition


MODEL_ID = "stabilityai/sdxl-turbo"
OUTPUT = Path(os.environ.get(
    "FURIOSA_REPRO_DIR", "reports/rngd_porting/runs/sdxl_downblock0_resnet0_repro"
))
OUTPUT.mkdir(parents=True, exist_ok=True)
os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = str(OUTPUT / "compiler_graphs.jsonl")
for stale_name in ("compiler_graphs.jsonl", "furiosa_trace.json", "torch_cpu_trace.json",
                   "auditor_events.jsonl", "device_samples.jsonl", "summary.json"):
    (OUTPUT / stale_name).unlink(missing_ok=True)

torch.manual_seed(20261102)
unet = UNet2DConditionModel.from_pretrained(
    MODEL_ID,
    subfolder="unet",
    torch_dtype=torch.bfloat16,
    variant="fp16",
    token=os.environ.get("HF_TOKEN"),
    local_files_only=True,
)
block = unet.down_blocks[0].resnets[0]
operator = os.environ.get("FURIOSA_REPRO_OP", "resnet")
if operator in ("norm1", "norm1-decomposed", "norm1-fn-decomposed"):
    module = block.norm1 if operator in ("norm1", "norm1-fn-decomposed") else GroupNormMeanSquare(block.norm1)
    component = f"unet.down_blocks.0.resnets.0.{operator}"
    reference = deepcopy(block.norm1)
elif operator == "resnet-decomposed":
    reference = deepcopy(block)
    block.norm1 = GroupNormMeanSquare(block.norm1)
    block.norm2 = GroupNormMeanSquare(block.norm2)
    module = block
    component = "unet.down_blocks.0.resnets.0.groupnorm_decomposed"
elif operator == "resnet-fn-decomposed":
    module = block
    component = "unet.down_blocks.0.resnets.0.groupnorm_fx_decomposed"
    reference = deepcopy(block)
elif operator in ("downblock", "downblock-fn-decomposed", "downblock-segmented"):
    module = unet.down_blocks[0]
    component = f"unet.down_blocks.0.{operator}"
    reference = deepcopy(module)
else:
    module = block
    component = "unet.down_blocks.0.resnets.0"
    reference = deepcopy(module)
sample = torch.randn(1, 320, 64, 64, dtype=torch.bfloat16)
temb = torch.randn(1, 1280, dtype=torch.bfloat16)
with torch.inference_mode():
    expected = reference(sample) if operator.startswith("norm1") else reference(sample, temb)
if operator in ("norm1-fn-decomposed", "resnet-fn-decomposed", "downblock-fn-decomposed", "downblock-segmented"):
    install_group_norm_decomposition()
module.to("furiosa:0")
if operator == "downblock-segmented":
    _segment_down_block_operators(module, "sdxl", 0)
    compiled = module
else:
    compiled = _NPUBlock(module, "sdxl", component)
auditor = NPUExecutionAuditor(OUTPUT, "sdxl-down-block-0-resnet-0")
try:
    with auditor:
        with auditor.span(f"model_component::{component}"):
            if operator == "downblock-segmented":
                actual = compiled(sample.to("furiosa:0"), temb.to("furiosa:0"))
            else:
                actual = compiled(sample) if operator.startswith("norm1") else compiled(sample, temb)
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)
    else:
        torch.utils._pytree.tree_map(
            lambda got, want: torch.testing.assert_close(got.cpu(), want, rtol=5e-2, atol=5e-2),
            actual, expected,
        )
    summary = json.loads((OUTPUT / "summary.json").read_text())
    if summary["counts"]["npu_trace_confirmed_graphs"] < 1:
        raise RuntimeError("No graph-correlated NPU Task event was recorded")
    result = {"status": "npu_and_numeric_pass", "shape": list(sample.shape),
              "dtype": str(sample.dtype), "npu_tasks": summary["profiler"]["npu_task_events"],
              "first_failure": None}
except Exception as exc:
    result = {"status": "failed", "shape": list(sample.shape), "dtype": str(sample.dtype),
              "error": str(exc).splitlines()[0],
              "summary": str(OUTPUT / "summary.json")}
    print(json.dumps(result), flush=True)
    raise
else:
    print(json.dumps(result), flush=True)
