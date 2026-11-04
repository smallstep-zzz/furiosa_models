"""Opt-in integration checks that need a live Furiosa RNGD device."""

import os
import json
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

from compiler_engine.renegade_diffusion_compiler import _audited_backend
from compiler_engine.graph_lowerings import install_device_safe_sdpa, install_dtype_promoting_cat
from compiler_engine.device_placements import install_qwen_rotary_device_fix


@unittest.skipUnless(os.environ.get("FURIOSA_RUN_NPU_TESTS") == "1", "set FURIOSA_RUN_NPU_TESTS=1 on a Furiosa server")
class NpuLoweringIntegrationTests(unittest.TestCase):
    def test_mixed_dtype_cat_runs_on_furiosa_with_pytorch_promotion(self):
        device = torch.device("furiosa:0")
        left = torch.ones((1, 4), dtype=torch.bfloat16)
        right = torch.full((1, 4), 2.0, dtype=torch.float32)
        expected = torch.cat([left, right], dim=-1)
        install_dtype_promoting_cat()
        actual = torch.concat([left.to(device), right.to(device)], dim=-1)
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.device.type, "furiosa")
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)

    def test_decomposed_sdpa_uses_furiosa_and_matches_torch(self):
        class Attention(torch.nn.Module):
            def forward(self, query, key, value, mask):
                return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

        generator = torch.Generator().manual_seed(17)
        query = torch.randn(1, 2, 16, 8, generator=generator)
        key = torch.randn(1, 2, 16, 8, generator=generator)
        value = torch.randn(1, 2, 16, 8, generator=generator)
        mask = torch.ones(16, 16, dtype=torch.bool)
        expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        device = torch.device("furiosa:0")
        compiled = torch.compile(Attention(), backend=_audited_backend("unit", "sdpa"))
        actual = compiled(query.to(device), key.to(device), value.to(device), mask.to(device))
        self.assertEqual(actual.device.type, "furiosa")
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)

    def test_cpu_created_attention_mask_is_normalized_before_dynamo(self):
        class Attention(torch.nn.Module):
            def forward(self, query, key, value):
                positions = torch.arange(query.shape[-2])
                mask = positions[:, None] <= positions[None, :]
                return F.scaled_dot_product_attention(query, key, value, attn_mask=mask)

        generator = torch.Generator().manual_seed(19)
        query = torch.randn(1, 2, 16, 8, generator=generator)
        key = torch.randn(1, 2, 16, 8, generator=generator)
        value = torch.randn(1, 2, 16, 8, generator=generator)
        positions = torch.arange(16)
        mask = positions[:, None] <= positions[None, :]
        expected = F.scaled_dot_product_attention(query, key, value, attn_mask=mask)
        install_device_safe_sdpa()
        device = torch.device("furiosa:0")
        compiled = torch.compile(Attention(), backend=_audited_backend("unit", "sdpa_cpu_mask"))
        actual = compiled(query.to(device), key.to(device), value.to(device))
        self.assertEqual(actual.device.type, "furiosa")
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)

    def test_qwen_rotary_factors_are_normalized_before_dynamo(self):
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

        original = qwen3_vl.apply_rotary_pos_emb
        generator = torch.Generator().manual_seed(23)
        q = torch.randn(1, 2, 16, 8, generator=generator)
        k = torch.randn(1, 2, 16, 8, generator=generator)
        cos = torch.randn(1, 16, 8, generator=generator)
        sin = torch.randn(1, 16, 8, generator=generator)
        expected = original(q, k, cos, sin)
        install_qwen_rotary_device_fix()
        device = torch.device("furiosa:0")
        actual = qwen3_vl.apply_rotary_pos_emb(q.to(device), k.to(device), cos, sin)
        self.assertEqual(actual[0].device.type, "furiosa")
        torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=2e-2, atol=2e-2)


    def test_mixed_cpu_rotary_inputs_are_normalized_for_backend(self):
        class Rotary(torch.nn.Module):
            def forward(self, query, key, cos, sin):
                cos = cos.to(query.device).unsqueeze(1)
                sin = sin.to(query.device).unsqueeze(1)
                return query * cos + query * 0 + key * sin

        generator = torch.Generator().manual_seed(29)
        query = torch.randn(1, 2, 16, 8, generator=generator)
        key = torch.randn(1, 2, 16, 8, generator=generator)
        cos = torch.randn(1, 16, 8, generator=generator)
        sin = torch.randn(1, 16, 8, generator=generator)
        expected = query * cos.unsqueeze(1) + query * 0 + key * sin.unsqueeze(1)
        device = torch.device("furiosa:0")
        compiled = torch.compile(Rotary(), backend=_audited_backend("unit", "rotary_mixed_inputs"))
        actual = compiled(query.to(device), key.to(device), cos, sin)
        self.assertEqual(actual.device.type, "furiosa")
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-2, atol=2e-2)


    def test_qwen_rotary_function_compiles_with_mixed_devices(self):
        from transformers.models.qwen3_vl import modeling_qwen3_vl as qwen3_vl

        install_qwen_rotary_device_fix()

        class Rotary(torch.nn.Module):
            def forward(self, query, key, cos, sin):
                return qwen3_vl.apply_rotary_pos_emb(query, key, cos, sin)

        generator = torch.Generator().manual_seed(31)
        query = torch.randn(1, 2, 16, 8, generator=generator)
        key = torch.randn(1, 2, 16, 8, generator=generator)
        cos = torch.randn(1, 16, 8, generator=generator)
        sin = torch.randn(1, 16, 8, generator=generator)
        expected = qwen3_vl.apply_rotary_pos_emb(query, key, cos, sin)
        device = torch.device("furiosa:0")
        compiled = torch.compile(Rotary(), backend=_audited_backend("unit", "qwen_rotary"))
        actual = compiled(query.to(device), key.to(device), cos, sin)
        self.assertEqual(actual[0].device.type, "furiosa")
        torch.testing.assert_close(actual[0].cpu(), expected[0], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(actual[1].cpu(), expected[1], rtol=2e-2, atol=2e-2)


    def test_tcl_conv2d_route_runs_on_furiosa_and_matches_torch(self):
        from compiler_engine.operator_compat import run_operator

        generator = torch.Generator().manual_seed(37)
        value = torch.randn(2, 3, 8, 8, dtype=torch.bfloat16, generator=generator)
        filt = torch.randn(5, 3, 3, 3, dtype=torch.bfloat16, generator=generator)
        bias = torch.randn(5, dtype=torch.bfloat16, generator=generator)
        expected = F.conv2d(value, filt, bias=bias, padding=1, stride=2)
        device = torch.device("furiosa:0")
        actual = run_operator("conv2d", value.to(device), filt.to(device), bias.to(device), backend="tcl")
        self.assertEqual(actual.device.type, "furiosa")
        torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)



    def test_tcl_sdxl_conv_in_channel_tiling_is_currently_unsupported(self):
        from compiler_engine.operator_compat import run_operator

        generator = torch.Generator().manual_seed(41)
        value = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, generator=generator)
        filt = torch.randn(320, 4, 3, 3, dtype=torch.bfloat16, generator=generator)
        bias = torch.randn(320, dtype=torch.bfloat16, generator=generator)
        device = torch.device("furiosa:0")
        with self.assertRaisesRegex(Exception, "failed to compile"):
            run_operator("conv2d", value.to(device), filt.to(device), bias.to(device), backend="tcl")


    def test_sdxl_conv_in_spatial_linear_taps_run_on_npu(self):
        from furiosa.torch.profiler import Profiler
        from compiler_engine.renegade_diffusion_compiler import _SDXLConvInSpatialLinear

        generator = torch.Generator().manual_seed(43)
        value = torch.randn(1, 4, 64, 64, dtype=torch.bfloat16, generator=generator)
        weight = torch.randn(320, 4, 3, 3, dtype=torch.bfloat16, generator=generator)
        bias = torch.randn(320, dtype=torch.bfloat16, generator=generator)
        from types import SimpleNamespace
        conv = torch.nn.Conv2d(4, 320, 3, stride=1, padding=1, dtype=torch.bfloat16)
        with torch.no_grad():
            conv.weight.copy_(weight)
            conv.bias.copy_(bias)
        expected = F.conv2d(value, weight, bias=bias, padding=1, stride=1)
        conv = conv.to("furiosa:0")
        decomposed = _SDXLConvInSpatialLinear(conv, "sdxl")

        import os

        out = Path("reports/rngd_porting/runs/sdxl_conv_in_spatial_linear_tcl")
        out.mkdir(parents=True, exist_ok=True)
        graph_path = out / "compiler_graphs.jsonl"
        trace_path = out / "furiosa_trace.json"
        previous_graph_path = os.environ.get("FURIOSA_GRAPH_AUDIT_PATH")
        os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = str(graph_path)
        try:
            with Profiler(str(trace_path)):
                actual = decomposed(value)
        finally:
            if previous_graph_path is None:
                os.environ.pop("FURIOSA_GRAPH_AUDIT_PATH", None)
            else:
                os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = previous_graph_path
        trace = json.loads(trace_path.read_text())
        tasks = [event for event in trace.get("traceEvents", [])
                 if "npu2::Task::npu_graph::" in str(event.get("name", ""))
                 and "unet.conv_in.spatial_tap" in str(event.get("name", ""))]
        self.assertEqual(actual.device.type, "furiosa")
        self.assertGreaterEqual(len(tasks), 18, "expected nine NPU tap graphs with per-cluster tasks")
        torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)
        (out / "summary.json").write_text(json.dumps({
            "input_shape": list(value.shape), "weight_shape": list(weight.shape),
            "dtype": str(value.dtype), "route": "nine independently compiled spatial linear taps",
            "output_device": str(actual.device), "cpu_reference_match": True,
            "graph_correlated_npu_tasks": len(tasks),
        }, indent=2) + "\n")

    def test_sdxl_conv_in_single_channel_linear_runs_on_npu(self):
        from furiosa.torch.profiler import Profiler

        generator = torch.Generator().manual_seed(53)
        x = torch.randn(1, 64, 64, 4, dtype=torch.bfloat16, generator=generator)
        module = torch.nn.Linear(4, 320, dtype=torch.bfloat16)
        expected = F.linear(x, module.weight.detach().clone(), module.bias.detach().clone())
        device = torch.device("furiosa:0")
        module = module.to(device)
        compiled = torch.compile(module, backend=_audited_backend("sdxl", "unet.conv_in.tap"))
        out = Path("reports/rngd_porting/runs/sdxl_conv_in_linear")
        out.mkdir(parents=True, exist_ok=True)
        trace_path = out / "furiosa_trace.json"
        with Profiler(str(trace_path)):
            actual = compiled(x.to(device))
        trace = json.loads(trace_path.read_text())
        tasks = [event for event in trace.get("traceEvents", [])
                 if "npu2::Task::npu_graph::" in str(event.get("name", ""))
                 and "unet.conv_in.tap" in str(event.get("name", ""))]
        self.assertEqual(actual.device.type, "furiosa")
        self.assertGreater(len(tasks), 0)
        torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)

    def test_sdxl_linear640_sequence_runs_on_npu(self):
        generator = torch.Generator().manual_seed(59)
        value = torch.randn(1, 1024, 640, dtype=torch.bfloat16, generator=generator)
        module = torch.nn.Linear(640, 640, dtype=torch.bfloat16)
        expected = F.linear(value, module.weight.detach().clone(), module.bias.detach().clone())
        compiled = torch.compile(
            module.to("furiosa:0"),
            backend=_audited_backend("sdxl", "feature.linear640_sequence"),
        )
        with torch.inference_mode():
            actual = compiled(value.to("furiosa:0"))
        self.assertEqual(actual.device.type, "furiosa")
        torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)

    def test_sdxl_spatial_conv640_taps_run_on_npu(self):
        from furiosa.torch.profiler import Profiler
        from compiler_engine.renegade_diffusion_compiler import _NPUConv2dSpatialTaps

        generator = torch.Generator().manual_seed(61)
        value = torch.randn(1, 640, 32, 32, dtype=torch.bfloat16, generator=generator)
        conv = torch.nn.Conv2d(640, 640, 3, padding=1, dtype=torch.bfloat16)
        expected = F.conv2d(value, conv.weight.detach().clone(),
                            conv.bias.detach().clone(), padding=1)
        conv = conv.to("furiosa:0")
        decomposed = _NPUConv2dSpatialTaps(conv, "sdxl", "feature.spatial_conv640")
        out = Path("reports/rngd_porting/runs/features/spatial_conv640_unit")
        out.mkdir(parents=True, exist_ok=True)
        trace_path = out / "furiosa_trace.json"
        with Profiler(str(trace_path)):
            with torch.inference_mode():
                actual = decomposed(value)
        trace = json.loads(trace_path.read_text())
        tasks = [event for event in trace.get("traceEvents", [])
                 if "npu2::Task::npu_graph::" in str(event.get("name", ""))
                 and "unet.spatial_conv.640x640x3x3" in str(event.get("name", ""))]
        self.assertEqual(actual.device.type, "furiosa")
        self.assertGreater(len(tasks), 0)
        torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)


    def test_sdxl_spatial_conv640_to_1280_fused_run_on_npu(self):
        from furiosa.torch.profiler import Profiler
        from compiler_engine.renegade_diffusion_compiler import _NPUConv2dSpatialTaps

        generator = torch.Generator().manual_seed(67)
        value = torch.randn(1, 640, 16, 16, dtype=torch.bfloat16, generator=generator)
        conv = torch.nn.Conv2d(640, 1280, 3, padding=1, dtype=torch.bfloat16)
        expected = F.conv2d(value, conv.weight.detach().clone(),
                            conv.bias.detach().clone(), padding=1)
        conv = conv.to("furiosa:0")
        decomposed = _NPUConv2dSpatialTaps(
            conv, "sdxl", "feature.spatial_conv640_to_1280"
        )
        out = Path("reports/rngd_porting/runs/features/spatial_conv640_to_1280_unit")
        out.mkdir(parents=True, exist_ok=True)
        trace_path = out / "furiosa_trace.json"
        with Profiler(str(trace_path)):
            with torch.inference_mode():
                actual = decomposed(value)
        trace = json.loads(trace_path.read_text())
        tasks = [event for event in trace.get("traceEvents", [])
                 if "npu2::Task::npu_graph::" in str(event.get("name", ""))
                 and "unet.spatial_conv.640x1280x3x3" in str(event.get("name", ""))]
        self.assertEqual(actual.device.type, "furiosa")
        self.assertGreater(len(tasks), 0)
        torch.testing.assert_close(actual.cpu(), expected, rtol=5e-2, atol=5e-2)

if __name__ == "__main__":
    unittest.main()
