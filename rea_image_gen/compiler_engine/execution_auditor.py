from pathlib import Path
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import datetime, timezone
import importlib.metadata
import json
import faulthandler
import os
import re
import subprocess
import threading
import time
import torch


def _json_default(value):
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return repr(value)


@contextmanager
def record_function(name):
    with ExitStack() as stack:
        try:
            from furiosa.torch.profiler import RecordFunction
            stack.enter_context(RecordFunction(name, propagate=True))
        except Exception:
            pass
        try:
            stack.enter_context(torch.profiler.record_function(name))
        except Exception:
            pass
        yield


def instrument_callable(owner, attribute, label):
    original = getattr(owner, attribute, None)
    if original is None or getattr(original, "_furiosa_audited", False):
        return False
    def wrapped(*args, **kwargs):
        with record_function(label):
            return original(*args, **kwargs)
    wrapped._furiosa_audited = True
    setattr(owner, attribute, wrapped)
    return True


def _attribute_npu_tasks(trace_events, task_events):
    from collections import Counter, defaultdict

    prefixes = ("model_component::", "host::", "compiler_compile::")
    spans = [event for event in trace_events
             if event.get("ph") == "X" and str(event.get("name", "")).startswith(prefixes)]
    by_scope = defaultdict(Counter)
    unattributed = Counter()
    graph_tasks = operator_tasks = 0
    for task in task_events:
        name = str(task.get("name", ""))
        graph_id = task.get("graph_id")
        is_operator = "::Task::aten::" in name
        graph_tasks += graph_id is not None
        operator_tasks += is_operator
        scope = None
        if graph_id:
            parts = name.split("::")
            if "npu_graph" in parts:
                index = parts.index("npu_graph")
                scope = ("model_component", parts[index + 2])
        if scope is None:
            ts = task.get("ts")
            if ts is not None:
                candidates = [span for span in spans
                              if span.get("ts", 0) <= ts <= span.get("ts", 0) + span.get("dur", 0)]
                candidates.sort(key=lambda span: span.get("dur", float("inf")))
                for span in candidates:
                    span_name = str(span.get("name", ""))
                    prefix = next((item for item in prefixes if span_name.startswith(item)), None)
                    if prefix:
                        scope = (prefix[:-2], span_name[len(prefix):])
                        break
        operator = name.split("::Task::aten::", 1)[1] if is_operator else "<compiled-graph>"
        if scope:
            by_scope[scope][operator] += 1
        else:
            unattributed[operator] += 1
    return {
        "graph_task_events": graph_tasks,
        "operator_task_events": operator_tasks,
        "attributed_task_events": sum(sum(counter.values()) for counter in by_scope.values()),
        "unattributed_task_events": sum(unattributed.values()),
        "by_scope": {
            f"{kind}::{name}": {"task_events": sum(counter.values()),
                                "operator_task_events": sum(value for op, value in counter.items()
                                                             if op != "<compiled-graph>"),
                                "operators": dict(counter)}
            for (kind, name), counter in by_scope.items()
        },
        "unattributed_operators": dict(unattributed),
    }


class NPUExecutionAuditor:
    """Collect official Furiosa Torch NPU traces, CPU op traces, and device snapshots."""

    def __init__(self, output_dir, model_name):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.trace_path = self.output_dir / "furiosa_trace.json"
        self.cpu_trace_path = self.output_dir / "torch_cpu_trace.json"
        self.events_path = self.output_dir / "auditor_events.jsonl"
        self.samples_path = self.output_dir / "device_samples.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stack = ExitStack()
        self._sampler = None
        self._torch_profiler = None
        self._profile_error = None
        self._stack_file = None
        self._old_profile_level = os.environ.get("FURIOSA_TORCH_PROFILE")
        self.events_path.write_text("")
        self.samples_path.write_text("")

    def record(self, event, **fields):
        entry = {
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "time_monotonic_ns": time.monotonic_ns(),
            "model": self.model_name,
            "event": event,
            **fields,
        }
        with self._lock, self.events_path.open("a") as stream:
            stream.write(json.dumps(entry, default=_json_default) + "\n")

    def span(self, label):
        return record_function(label)

    def _sample_once(self, phase):
        sample = {"time_utc": datetime.now(timezone.utc).isoformat(), "phase": phase,
                  "process_pid": os.getpid()}
        for command, key in ((["furiosa-smi", "status"], "smi_status"),
                             (["furiosa-smi", "ps"], "smi_processes")):
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=5)
                sample[key] = result.stdout.strip()
                if result.returncode:
                    sample[key + "_error"] = result.stderr.strip()[:500]
            except Exception as exc:
                sample[key + "_error"] = repr(exc)
        try:
            import furiosa_smi
            devices = []
            for device in furiosa_smi.list_devices():
                devices.append({
                    "name": device.device_name(),
                    "uuid": device.uuid(),
                    "arch": str(device.arch()),
                    "liveness": str(device.liveness()),
                    "firmware": str(device.firmware_version()),
                    "memory": {
                        "total_bytes": sum(info.total_bytes() for info in device.memory_usage().values()),
                        "in_use_bytes": sum(info.in_use_bytes() for info in device.memory_usage().values()),
                    },
                    "core_status": {str(core): str(status) for core, status in device.core_status().items()},
                    "power_w": device.power_consumption(),
                    "temperature": str(device.device_temperature()),
                })
            sample["smi_python_devices"] = devices
            sample["driver"] = str(furiosa_smi.driver_info())
        except Exception as exc:
            sample["smi_python_error"] = repr(exc)
        with self._lock, self.samples_path.open("a") as stream:
            stream.write(json.dumps(sample, default=_json_default) + "\n")

    def _sample_loop(self):
        while not self._stop.wait(1.0):
            self._sample_once("during")

    def __enter__(self):
        self._sample_once("before")
        stack_timeout = float(os.environ.get("FURIOSA_AUDIT_STACK_TIMEOUT_SEC", "60"))
        self._stack_file = (self.output_dir / "cpu_stacks.log").open("a")
        faulthandler.dump_traceback_later(stack_timeout, repeat=True, file=self._stack_file)
        os.environ["FURIOSA_TORCH_PROFILE"] = "trace"
        try:
            from furiosa.torch.profiler import Profiler
            self._stack.enter_context(Profiler(str(self.trace_path)))
        except Exception as exc:
            self._profile_error = repr(exc)
            self.record("npu_profiler_unavailable", error=self._profile_error)
        try:
            self._torch_profiler = torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                record_shapes=True,
                with_stack=False,
                acc_events=True,
            )
            self._stack.enter_context(self._torch_profiler)
        except Exception as exc:
            self.record("torch_cpu_profiler_unavailable", error=repr(exc))
            self._torch_profiler = None
        self._sampler = threading.Thread(target=self._sample_loop, name="npu-audit-smi", daemon=True)
        self._sampler.start()
        self.record("audit_started", trace_path=str(self.trace_path),
                    cpu_trace_path=str(self.cpu_trace_path),
                    npu_profiler_error=self._profile_error)
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc is not None:
            message = str(exc).splitlines()[0][:500]
            event = "diagnostic_stop" if message.startswith("diagnostic stop after ") else "inference_exception"
            self.record(event, exception_type=getattr(exc_type, "__name__", str(exc_type)), message=message)
        self._stop.set()
        faulthandler.cancel_dump_traceback_later()
        if self._stack_file:
            self._stack_file.flush()
        if self._sampler:
            self._sampler.join(timeout=6)
        try:
            self._stack.__exit__(exc_type, exc, tb)
        finally:
            if self._torch_profiler is not None:
                try:
                    self._torch_profiler.export_chrome_trace(str(self.cpu_trace_path))
                except Exception as export_exc:
                    self.record("torch_cpu_trace_export_error", error=repr(export_exc))
            if self._stack_file:
                self._stack_file.close()
            if self._old_profile_level is None:
                os.environ.pop("FURIOSA_TORCH_PROFILE", None)
            else:
                os.environ["FURIOSA_TORCH_PROFILE"] = self._old_profile_level
            self._sample_once("after")
            self._write_summary()
        return False

    @classmethod
    def summarize_existing(cls, output_dir, model_name):
        auditor = cls.__new__(cls)
        auditor.output_dir = Path(output_dir)
        auditor.model_name = model_name
        auditor.trace_path = auditor.output_dir / "furiosa_trace.json"
        auditor.cpu_trace_path = auditor.output_dir / "torch_cpu_trace.json"
        auditor.events_path = auditor.output_dir / "auditor_events.jsonl"
        auditor.samples_path = auditor.output_dir / "device_samples.jsonl"
        auditor.summary_path = auditor.output_dir / "summary.json"
        auditor._write_summary()
        return auditor.summary_path

    def _write_summary(self):
        try:
            trace = json.loads(self.trace_path.read_text())
            npu_events = [
                event for event in trace.get("traceEvents", [])
                if "::Task::" in str(event.get("name", ""))
            ]
            task_events = []
            for event in npu_events:
                task_events.append({
                    "name": event.get("name"),
                    "duration_us": event.get("dur"),
                    "ts": event.get("ts"),
                    "graph_id": next((part for part in str(event.get("name", "")).split("::")
                                      if re.fullmatch(r"[0-9a-f]{16}", part)), None),
                    "execution_index": next((int(match.group(1)) for match in
                                             [re.search(r"execution-(\d+)", str(event.get("name", "")))]
                                             if match), None),
                    "compile_instance_id": next((match.group(1) for match in
                                                  [re.search(r"compile-([0-9]+-[0-9]+)", str(event.get("name", "")))]
                                                  if match), None),
                    "args": event.get("args", {}),
                })
        except Exception as exc:
            trace = {}
            task_events = []
            trace_error = repr(exc)
        else:
            trace_error = None
        compiler_path = Path(os.environ.get("FURIOSA_GRAPH_AUDIT_PATH", self.output_dir / "compiler_graphs.jsonl"))
        compiler_rows = []
        if compiler_path.is_file():
            compiler_rows = [json.loads(line) for line in compiler_path.read_text().splitlines() if line.strip()]
        final = {}
        for row in compiler_rows:
            if row.get("graph_id"):
                final[(row.get("component"), row["graph_id"], row.get("compile_instance_id"))] = row
        graphs = []
        for (component, graph_id, compile_instance_id), row in final.items():
            matches = [event for event in task_events
                       if event.get("graph_id") == graph_id and component in str(event.get("name", ""))
                       and (compile_instance_id is None or event.get("compile_instance_id") == compile_instance_id)]
            phase_rows = [r for r in compiler_rows if r.get("graph_id") == graph_id and
                          r.get("component") == component and r.get("compile_instance_id") == compile_instance_id]
            placements = [r.get("placement") for r in phase_rows]
            cpu_confirmed = "cpu" in placements or any(
                r.get("phase", "").startswith("cpu_fallback_") for r in phase_rows
            )
            if matches and cpu_confirmed:
                verdict = "mixed_npu_and_cpu_execution_observed"
            elif matches:
                verdict = "npu_confirmed_by_runtime_trace"
            elif cpu_confirmed:
                verdict = "cpu_model_op_confirmed"
            elif any(r.get("phase") == "npu_execution_failed" for r in phase_rows):
                verdict = "npu_task_observed_execution_failed" if matches else "execution_failed_before_correlated_npu_task"
            elif any(r.get("phase", "").endswith("execution_completed") for r in phase_rows):
                verdict = "unknown_execution_returned_without_correlated_npu_trace"
            elif any(r.get("phase", "").endswith("execution_start") for r in phase_rows):
                verdict = "unknown_execution_not_returned"
            elif any(r.get("placement") == "compile_failed" or r.get("phase") == "backend_compile_failed" for r in phase_rows):
                verdict = "compile_failed_before_execution"
            elif any(r.get("phase") == "backend_compile_start" for r in phase_rows):
                verdict = "compile_started_no_backend_return"
            elif any(r.get("phase") == "backend_returned" for r in phase_rows):
                verdict = "backend_returned_execution_not_submitted"
            else:
                verdict = "graph_capture_only_unknown"
            graphs.append({
                "graph_id": graph_id,
                "compile_instance_id": compile_instance_id,
                "component": component,
                "nodes": row.get("nodes"),
                "original_operators": row.get("operators", []),
                "lowered_operators": row.get("lowered_operators", []),
                "operator_mapping": None,
                "operator_mapping_note": "Runtime task trace is graph-correlated; fusion prevents a proven one-to-one FX-op mapping.",
                "inputs": row.get("inputs", []),
                "verdict": verdict,
                "phase_rows": len(phase_rows),
                "npu_task_count": len(matches),
                "npu_task_duration_us": sum(e.get("duration_us") or 0 for e in matches),
                "npu_tasks_by_execution": [
                    {"execution_index": index,
                     "count": sum(e.get("execution_index") == index for e in matches),
                     "duration_us": sum(e.get("duration_us") or 0 for e in matches if e.get("execution_index") == index)}
                    for index in sorted({e.get("execution_index") for e in matches if e.get("execution_index") is not None})
                ],
                "execution_prepare_durations_ms": [r.get("duration_ns", 0) / 1e6 for r in phase_rows if r.get("phase") == "execution_prepare_returned"],
                "first_execution_durations_ms": [r.get("duration_ns", 0) / 1e6 for r in phase_rows if r.get("phase") == "first_execution_completed"],
                "repeat_execution_durations_ms": [r.get("duration_ns", 0) / 1e6 for r in phase_rows if r.get("phase") == "repeat_execution_completed"],
                "last_phase": next((r.get("phase") for r in reversed(phase_rows) if r.get("phase")), None),
                "execution_error": next((r.get("error") or r.get("compile_error") or r.get("message") for r in reversed(phase_rows)
                                          if r.get("phase") == "npu_execution_failed"), None),
                "last_placement": row.get("placement"),
                "backend_compile_durations_ms": [r.get("duration_ns", 0) / 1e6 for r in phase_rows if r.get("phase") == "backend_returned"],
                "execution_durations_ms": [r.get("duration_ns", 0) / 1e6 for r in phase_rows
                                           if r.get("phase", "").endswith("execution_completed")],
                "repeat_execution_count": sum(r.get("phase") == "repeat_execution_completed" for r in phase_rows),
            })
        try:
            cpu_trace = json.loads(self.cpu_trace_path.read_text())
            cpu_events = cpu_trace.get("traceEvents", [])
        except Exception:
            cpu_events = []
        cpu_ops = [e for e in cpu_events if e.get("cat") == "cpu_op"]
        from collections import Counter
        annotations = [e for e in cpu_events if e.get("cat") in ("user_annotation", "annotation")]
        attributed = {
            "cpu_model_component": 0,
            "cpu_model_context_unclassified": 0,
            "cpu_host_orchestration": 0,
            "cpu_compiler_host": 0,
            "cpu_unattributed": 0,
        }
        # Sweep annotations per thread instead of O(ops * annotations) rescans.
        from collections import defaultdict
        import heapq
        spans_by_tid = defaultdict(list)
        for index, span in enumerate(annotations):
            spans_by_tid[span.get("tid")].append((span.get("ts", 0), index, span))
        for spans in spans_by_tid.values():
            spans.sort(key=lambda item: (item[0], item[1]))
        cursors = defaultdict(int)
        heaps = defaultdict(list)
        for op in sorted(cpu_ops, key=lambda e: e.get("ts", 0)):
            ts = op.get("ts", 0)
            tid = op.get("tid")
            spans = spans_by_tid[tid]
            cursor = cursors[tid]
            heap = heaps[tid]
            while cursor < len(spans) and spans[cursor][0] <= ts:
                begin, index, span = spans[cursor]
                duration = span.get("dur", float("inf"))
                heapq.heappush(heap, (duration, index, begin + span.get("dur", 0), span))
                cursor += 1
            cursors[tid] = cursor
            while heap and heap[0][2] < ts:
                heapq.heappop(heap)
            active = [item[3] for item in sorted(heap) if item[2] >= ts]
            category = None
            for span in active:
                name = str(span.get("name", ""))
                if name.startswith(("cpu_model_component::", "cpu_fallback::")):
                    category = "cpu_model_component"
                elif name.startswith("host::"):
                    category = "cpu_host_orchestration"
                elif name.startswith("compiler_compile::") or "compile" in name.lower() or "tracing" in name.lower():
                    category = "cpu_compiler_host"
                elif name.startswith(("model_component::", "npu_graph::")):
                    category = "cpu_model_context_unclassified"
                if category:
                    break
            attributed[category or "cpu_unattributed"] += 1
        cpu_summary = {
            "trace_available": bool(cpu_events),
            "aten_cpu_op_count": len(cpu_ops),
            "top_aten_cpu_ops": dict(Counter(e.get("name", "?") for e in cpu_ops).most_common(20)),
            "attributed_aten_cpu_ops": attributed,
            "model_cpu_fallback_graphs": sum(g["verdict"] in ("cpu_model_op_confirmed", "mixed_npu_and_cpu_execution_observed") for g in graphs),
            "host_vs_model_cpu_attribution": "CPU ATen ops under cpu_model_component/cpu_fallback spans are model CPU; ops under host spans are CPU host work; compilation and unattributed ops are separated or unknown. A model_component span alone is not enough to call it CPU inference.",
        }
        sdk_versions = {}
        for name in ("furiosa-torch", "furiosa-kernels", "furiosa-kernels-contrib", "furiosa-tcc"):
            try:
                sdk_versions[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        try:
            samples = [json.loads(line) for line in self.samples_path.read_text().splitlines() if line.strip()]
        except Exception:
            samples = []
        device_metrics = []
        for sample in samples:
            status = sample.get("smi_status", "")
            util_values = [float(value) for value in re.findall(r"Core \d+: ([0-9]+(?:\.[0-9]+)?)%", status)]
            memory_match = re.search(r"([0-9.]+)/([0-9.]+) GiB", status)
            processes = sample.get("smi_processes", "")
            for device in sample.get("smi_python_devices", []):
                device_metrics.append({
                    "phase": sample.get("phase"),
                    "process_pid": sample.get("process_pid"),
                    "process_mapped_to_npu": str(sample.get("process_pid")) in processes,
                    "name": device.get("name"),
                    "memory": device.get("memory"),
                    "cli_memory_gib": list(map(float, memory_match.groups())) if memory_match else None,
                    "core_utilization_pct": util_values,
                    "power_w": device.get("power_w"),
                    "temperature": device.get("temperature"),
                    "core_status": device.get("core_status"),
                })
        blockers = [graph for graph in graphs if graph["verdict"] in (
            "unknown_execution_not_returned", "compile_started_no_backend_return",
            "compile_failed_before_execution", "unknown_execution_returned_without_correlated_npu_trace",
            "execution_failed_before_correlated_npu_task", "npu_task_observed_execution_failed",
        )]
        audit_events = []
        try:
            audit_events = [json.loads(line) for line in self.events_path.read_text().splitlines() if line.strip()]
        except Exception:
            pass
        first_failure_point = None
        if blockers:
            graph = blockers[0]
            first_failure_point = {
                "component": graph["component"],
                "graph_id": graph["graph_id"],
                "verdict": graph["verdict"],
                "last_phase": graph["last_phase"],
                "execution_error": graph.get("execution_error"),
                "operators": graph["original_operators"],
                "inputs": graph["inputs"],
            }
        elif audit_events:
            exception = next((event for event in reversed(audit_events) if event.get("event") == "inference_exception"), None)
            if exception:
                first_failure_point = {"stage": "pipeline", "exception_type": exception.get("exception_type"),
                                       "message": exception.get("message")}
        summary = {
            "model": self.model_name,
            "sdk_versions": sdk_versions,
            "device": {"fusion_pes": os.environ.get("FURIOSA_FUSION_PES", "8"), "mesh": "Single"},
            "profiler": {"official_api": "furiosa.torch.profiler.Profiler", "profile_level": "trace",
                         "trace_available": self.trace_path.is_file(), "trace_error": trace_error,
                         "npu_task_events": len(task_events), "raw_trace_path": self.trace_path.name},
            "npu_execution": _attribute_npu_tasks(trace.get("traceEvents", []), task_events),
            "counts": {
                "graphs_seen": len({(g["component"], g["graph_id"]) for g in graphs}),
                "compile_instances_seen": len(graphs),
                "npu_trace_confirmed_graphs": len({(g["component"], g["graph_id"]) for g in graphs if g["verdict"] == "npu_confirmed_by_runtime_trace"}),
                "npu_trace_confirmed_compile_instances": sum(g["verdict"] == "npu_confirmed_by_runtime_trace" for g in graphs),
                "cpu_confirmed_model_graphs": sum(g["verdict"] in ("cpu_model_op_confirmed", "mixed_npu_and_cpu_execution_observed") for g in graphs),
                "unknown_graphs": sum(g["verdict"].startswith("unknown") or g["verdict"] == "compiled_or_pending_unknown" for g in graphs),
                "compile_failed_graphs": sum(g["verdict"] == "compile_failed_before_execution" for g in graphs),
                "execution_failed_graphs": sum(g["verdict"] == "execution_failed_before_correlated_npu_task" for g in graphs),
                "partial_npu_execution_failures": sum(g["verdict"] == "npu_task_observed_execution_failed" for g in graphs),
                "cpu_aten_ops": len(cpu_ops),
            },
            "cpu": cpu_summary,
            "graphs": graphs,
            "first_failure_point": first_failure_point,
            "intentional_diagnostic_stop": next((event.get("message") for event in reversed(audit_events)
                                                 if event.get("event") == "diagnostic_stop"), None),
            "auditor_events_path": self.events_path.name,
            "device_samples_path": self.samples_path.name,
            "device_metrics_samples": device_metrics,
            "compiler_events_path": compiler_path.name,
            "trace_event_examples": task_events[:12],
            "trace_parse_error": trace_error,
        }
        self.summary_path.write_text(json.dumps(summary, indent=2, default=_json_default) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Summarize a complete or interrupted RNGD execution audit.")
    parser.add_argument("output_dir")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    print(NPUExecutionAuditor.summarize_existing(args.output_dir, args.model))
