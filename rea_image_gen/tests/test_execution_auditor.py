from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import unittest
from unittest.mock import patch

import torch

from compiler_engine.execution_auditor import NPUExecutionAuditor


class AuditorVerdictTests(unittest.TestCase):
    def _summary(self, compiler_rows, trace_events):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            compiler = root / "compiler_graphs.jsonl"
            compiler.write_text("".join(json.dumps(row) + "\n" for row in compiler_rows))
            with patch.dict(os.environ, {"FURIOSA_GRAPH_AUDIT_PATH": str(compiler)}):
                auditor = NPUExecutionAuditor(root, "unit")
                auditor.trace_path.write_text(json.dumps({"traceEvents": trace_events}))
                auditor.cpu_trace_path.write_text(json.dumps({"traceEvents": []}))
                auditor._write_summary()
                return json.loads(auditor.summary_path.read_text())

    def test_npu_requires_task_event_correlated_to_graph_and_component(self):
        row = {"component": "unet.block", "graph_id": "0123456789abcdef",
               "nodes": 3, "operators": ["linear"], "lowered_operators": ["linear"],
               "placement": "npu_compiled", "phase": "first_execution_completed"}
        trace = [{"name": "npu2::Task::npu_graph::sdxl::unet.block::0123456789abcdef::execution-1",
                  "dur": 20}]
        summary = self._summary([row], trace)
        self.assertEqual(summary["counts"]["npu_trace_confirmed_graphs"], 1)
        self.assertEqual(summary["graphs"][0]["npu_tasks_by_execution"][0]["execution_index"], 1)

    def test_recompiled_graph_tasks_match_the_compile_instance(self):
        rows = [
            {"component": "unet.block", "graph_id": "0123456789abcdef",
             "compile_instance_id": instance, "nodes": 3,
             "operators": ["linear"], "lowered_operators": ["linear"],
             "placement": "npu_compiled", "phase": "backend_returned"}
            for instance in ("123-100", "123-200")
        ]
        trace = [{"name": "npu2::Task::npu_graph::unit::unet.block::0123456789abcdef::"
                          "compile-123-200::execution-1", "dur": 20}]
        summary = self._summary(rows, trace)
        by_compile = {graph["compile_instance_id"]: graph for graph in summary["graphs"]}
        self.assertEqual(by_compile["123-200"]["verdict"], "npu_confirmed_by_runtime_trace")
        self.assertEqual(by_compile["123-100"]["verdict"], "backend_returned_execution_not_submitted")

    def test_direct_npu_operator_tasks_are_attributed_by_runtime_spans(self):
        trace = [
            {"name": "model_component::text_encoder_2", "ph": "X", "ts": 100, "dur": 500},
            {"name": "host::prompt_encode", "ph": "X", "ts": 700, "dur": 100},
            {"name": "npu2::Task::aten::linear", "ph": "X", "ts": 200, "dur": 10},
            {"name": "npu2::Task::aten::arange", "ph": "X", "ts": 720, "dur": 4},
            {"name": "npu2::Task::aten::gelu", "ph": "X", "ts": 900, "dur": 4},
        ]
        summary = self._summary([], trace)
        npu = summary["npu_execution"]
        self.assertEqual(npu["operator_task_events"], 3)
        self.assertEqual(npu["by_scope"]["model_component::text_encoder_2"]["operators"]["linear"], 1)
        self.assertEqual(npu["by_scope"]["host::prompt_encode"]["operators"]["arange"], 1)
        self.assertEqual(npu["unattributed_operators"]["gelu"], 1)

    def test_compile_or_execution_stall_stays_unknown(self):
        row = {"component": "unet.conv_in", "graph_id": "0123456789abcdef",
               "nodes": 3, "operators": ["conv2d"], "lowered_operators": ["conv2d"],
               "placement": "npu_execution_started", "phase": "first_execution_start"}
        summary = self._summary([row], [])
        self.assertEqual(summary["graphs"][0]["verdict"], "unknown_execution_not_returned")

    def test_reported_execution_failure_is_not_left_unknown(self):
        row = {"component": "unet.conv_in", "graph_id": "0123456789abcdef",
               "nodes": 3, "operators": ["conv2d"], "lowered_operators": ["conv2d"],
               "placement": "npu_execution_failed", "phase": "npu_execution_failed",
               "error": "device execution failed"}
        summary = self._summary([row], [])
        self.assertEqual(summary["graphs"][0]["verdict"], "execution_failed_before_correlated_npu_task")
        self.assertEqual(summary["graphs"][0]["execution_error"], "device execution failed")
        self.assertEqual(summary["counts"]["execution_failed_graphs"], 1)

    def test_nested_host_work_takes_precedence_over_model_span(self):
        with TemporaryDirectory() as temporary:
            auditor = NPUExecutionAuditor(temporary, "nested_host")
            with auditor:
                with auditor.span("model_component::unet.conv_in"):
                    with auditor.span("host::unet.conv_in.window_preparation"):
                        value = torch.arange(4)
            summary = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertGreater(summary["cpu"]["attributed_aten_cpu_ops"]["cpu_host_orchestration"], 0)
            self.assertEqual(summary["cpu"]["attributed_aten_cpu_ops"]["cpu_model_context_unclassified"], 0)

    def test_nested_model_work_takes_precedence_over_prompt_host_span(self):
        with TemporaryDirectory() as temporary:
            auditor = NPUExecutionAuditor(temporary, "nested_model")
            with auditor:
                with auditor.span("host::prompt_encode"):
                    with auditor.span("model_component::text_encoder.layer"):
                        value = torch.arange(4)
            summary = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertGreater(summary["cpu"]["attributed_aten_cpu_ops"]["cpu_model_context_unclassified"], 0)
            self.assertEqual(summary["cpu"]["attributed_aten_cpu_ops"]["cpu_host_orchestration"], 0)

    def test_cpu_fallback_is_separate_from_host_cpu_work(self):
        row = {"component": "unet.conv_in", "graph_id": "0123456789abcdef",
               "nodes": 3, "operators": ["conv2d"], "lowered_operators": ["conv2d"],
               "placement": "cpu", "phase": "cpu_fallback_completed"}
        summary = self._summary([row], [])
        self.assertEqual(summary["counts"]["cpu_confirmed_model_graphs"], 1)


class AuditorControlTests(unittest.TestCase):
    def test_cpu_only_linear_is_negative_control(self):
        with TemporaryDirectory() as temporary:
            auditor = NPUExecutionAuditor(temporary, "negative_cpu_linear")
            with auditor:
                with auditor.span("cpu_model_component::negative_cpu_linear"):
                    result = torch.nn.functional.linear(torch.ones(2, 4), torch.ones(3, 4))
            self.assertEqual(result.device.type, "cpu")
            summary = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertEqual(summary["profiler"]["npu_task_events"], 0)
            self.assertGreater(summary["cpu"]["attributed_aten_cpu_ops"]["cpu_model_component"], 0)

    def test_host_cpu_tensor_preparation_is_not_model_fallback(self):
        with TemporaryDirectory() as temporary:
            auditor = NPUExecutionAuditor(temporary, "host_tokenizer_control")
            with auditor:
                with auditor.span("host::tokenizer_control"):
                    tokens = torch.arange(8)
                    mask = tokens < 4
            self.assertEqual(int(mask.sum()), 4)
            summary = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertEqual(summary["profiler"]["npu_task_events"], 0)
            self.assertGreater(summary["cpu"]["attributed_aten_cpu_ops"]["cpu_host_orchestration"], 0)
            self.assertEqual(summary["cpu"]["model_cpu_fallback_graphs"], 0)


@unittest.skipUnless(os.environ.get("FURIOSA_RUN_NPU_TESTS") == "1",
                     "set FURIOSA_RUN_NPU_TESTS=1 on a Furiosa server")
class NpuAuditorPositiveControl(unittest.TestCase):
    def test_compiled_relu_requires_correlated_npu_task(self):
        from compiler_engine.renegade_diffusion_compiler import _audited_backend
        with TemporaryDirectory() as temporary:
            graph_path = str(Path(temporary) / "compiler_graphs.jsonl")
            os.environ["FURIOSA_GRAPH_AUDIT_PATH"] = graph_path
            os.environ["FURIOSA_REQUIRE_NPU"] = "1"
            compiled = torch.compile(torch.nn.ReLU(), backend=_audited_backend("auditor", "positive_relu"))
            value = torch.tensor([-1.0, 2.0], dtype=torch.bfloat16).to("furiosa:0")
            auditor = NPUExecutionAuditor(temporary, "positive_relu")
            with auditor:
                with auditor.span("model_component::positive_relu"):
                    result = compiled(value)
            torch.testing.assert_close(result.cpu(), torch.tensor([0.0, 2.0], dtype=torch.bfloat16))
            summary = json.loads((Path(temporary) / "summary.json").read_text())
            self.assertGreater(summary["counts"]["npu_trace_confirmed_graphs"], 0)


if __name__ == "__main__":
    unittest.main()
