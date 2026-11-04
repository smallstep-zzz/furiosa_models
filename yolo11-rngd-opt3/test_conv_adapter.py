"""CPU self-check; --npu is exclusively for the coordinator's shared queue."""
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("RAYON_NUM_THREADS", "4")
import numpy as np
import torch
from conv_adapter import SpatialLinearConv2d


def compare(actual, expected):
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    delta = (actual - expected).abs()
    return {"max_abs": float(delta.max()), "rmse": float(delta.square().mean().sqrt()),
            "nonidentical": int(torch.count_nonzero(delta)), "elements": expected.numel()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--npu", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20261103)
    started = time.perf_counter()
    synthetic = []
    with torch.no_grad():
        for shape, kernel, stride, padding, bias in [
            ((1, 3, 640, 640), 3, 2, 1, True),
            ((2, 3, 9, 10), 3, 2, 1, False),
            ((1, 3, 8, 7), (2, 3), (1, 2), (1, 0), True),
            ((1, 3, 3, 3), 3, 2, 1, True),
        ]:
            conv = torch.nn.Conv2d(3, 16, kernel, stride, padding, bias=bias).eval()
            x = torch.randn(shape)
            synthetic.append(compare(SpatialLinearConv2d(conv)(x), conv(x)))
        # Unit impulses exercise all nine tap positions and zero-padded borders.
        conv = torch.nn.Conv2d(3, 16, 3, 2, 1).eval()
        x = torch.zeros(1, 3, 7, 8)
        x[:, :, 0, 0] = 1
        x[:, :, -1, -1] = -1
        synthetic.append(compare(SpatialLinearConv2d(conv)(x), conv(x)))
        import onnx
        from onnx2torch import convert
        model = convert(onnx.load(args.root / "models/yolo11n_640.onnx")).eval()
        conv = next(m for m in model.modules() if isinstance(m, torch.nn.Conv2d))
        assert tuple(conv.weight.shape) == (16, 3, 3, 3)
        assert conv.stride == (2, 2) and conv.padding == (1, 1)
        x = torch.from_numpy(np.load(args.root / "cpu_reference.npz")["input"])
        assert tuple(x.shape) == (1, 3, 640, 640) and x.dtype == torch.float32
        candidate = SpatialLinearConv2d(conv).eval()
        expected = conv(x)
        report = {"cpu": compare(candidate(x), expected), "synthetic": synthetic,
                  "cpu_s": time.perf_counter() - started, "npu": "untested"}
        if args.npu:
            import fcntl
            import sys
            os.environ["FURIOSA_REQUIRE_NPU"] = "1"
            sys.path.insert(0, "/root/rngd-model-forge")
            from furiosa.torch import CompileModule
            from furiosa.torch.config import CompilerConfig, TacticHintConfig, set_cpu_fallback
            from compiler_engine.renegade_diffusion_compiler import _configure_fused_pes
            from compiler_engine.execution_auditor import NPUExecutionAuditor
            set_cpu_fallback(False)
            out = args.root / "runs" / ("conv_spatial_linear_" + str(time.time_ns()))
            out.mkdir(parents=True)
            lock = open("/root/rngd-model-forge/reports/rngd_porting/npu.lock", "a")
            fcntl.flock(lock, fcntl.LOCK_EX)
            _configure_fused_pes()
            config = CompilerConfig(tactic_hint=TacticHintConfig.ForVisionModel,
                                    implicit_type_casting=False)
            started = time.perf_counter()
            # Compile only contraction so packing cannot fuse back into spatial Conv2d.
            packed_example = candidate.pack(x)
            class Contraction(torch.nn.Module):
                def __init__(self, linear):
                    super().__init__()
                    self.linear = linear
                def forward(self, value):
                    return self.linear(value)
            compiled = CompileModule.from_module(Contraction(candidate.linear), (packed_example,),
                                                 compiler_config=config).to("furiosa:0")
            report["compile_prepare_s"] = time.perf_counter() - started
            resident = x.to("furiosa:0")
            with NPUExecutionAuditor(out / "audit", "yolo11-first-conv-spatial-linear") as audit:
                with audit.span("model_component::first_conv"):
                    packed = candidate.pack(resident)
                    assert packed.device.type == "furiosa"
                    result = candidate.unpack(compiled(packed), resident.shape)
                    assert result.device.type == "furiosa"
                    result = result.cpu()  # Diagnostic output boundary only.
            report["npu_numeric"] = compare(result, expected)
            summary = json.loads((out / "audit/summary.json").read_text())
            report["npu_tasks"] = summary["profiler"]["npu_task_events"]
            assert report["npu_tasks"] > 0
            report["npu"] = "numeric_and_tasks_passed_requires_cpu_route_audit"
            report["evidence"] = str(out)
            (out / "result.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
