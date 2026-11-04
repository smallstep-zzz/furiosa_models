"""Compile Diffusers graphs with Furiosa and audit NPU/CPU placement."""

import hashlib
import json
import multiprocessing
import os
import time
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils import _pytree
from torch.nn.attention import SDPBackend, sdpa_kernel
import furiosa.torch as ft
from furiosa.torch import MeshKind, get_fusion, set_fusion, set_mesh_kind
from furiosa.torch.config import CompilerConfig, TacticHintConfig

from .graph_lowerings import (
    lower_scaled_dot_product_attention, install_device_safe_sdpa,
    install_dtype_promoting_cat, install_group_norm_decomposition,
    manual_scaled_dot_product_attention,
)
from .device_fallback import replace_graph_device
from .device_placements import install_qwen_rotary_device_fix
from .execution_auditor import record_function as _record_function


_THREADS = max(1, multiprocessing.cpu_count() - 2)
os.environ.setdefault("RAYON_NUM_THREADS", str(_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", str(_THREADS))
_FUSION_PES = int(os.environ.get("FURIOSA_FUSION_PES", "8"))
if not 1 <= _FUSION_PES <= 8:
    raise ValueError(f"FURIOSA_FUSION_PES must be between 1 and 8, got {_FUSION_PES}")
set_fusion(_FUSION_PES)
set_mesh_kind(MeshKind.Single())
_NPU_DEVICE = torch.device("furiosa:0")
_FUSION_CONFIGURED = True
_COMPILER_CONFIG = CompilerConfig(tactic_hint=TacticHintConfig.ForVisionModel)
_STRICT_BACKEND = ft.backend.with_config(_COMPILER_CONFIG, eager_fallback=False)
_NPU_EAGER_BACKEND = ft.backend.with_config(_COMPILER_CONFIG, eager_fallback=True)
_COMPONENTS = ("transformer", "unet", "vae", "text_encoder", "text_encoder_2")


def _require_npu_execution():
    return os.environ.get("FURIOSA_REQUIRE_NPU") == "1"


def _move_tensors(value, device):
    return _pytree.tree_map(
        lambda item: item.to(device)
        if isinstance(item, torch.Tensor) and item.device != device
        else item,
        value,
    )


def _normalize_mixed_float_inputs(args):
    tensors = [item for item in args if isinstance(item, torch.Tensor)]
    has_npu = any(item.device.type == _NPU_DEVICE.type for item in tensors)
    has_cpu = any(item.device.type == "cpu" for item in tensors)
    if not (has_npu and has_cpu):
        return args
    return tuple(
        item.to(_NPU_DEVICE)
        if isinstance(item, torch.Tensor)
        and item.device.type == "cpu"
        and item.is_floating_point()
        else item
        for item in args
    )


def _configure_fused_pes():
    global _FUSION_CONFIGURED
    if not _FUSION_CONFIGURED:
        set_fusion(_FUSION_PES)
        set_mesh_kind(MeshKind.Single())
        _FUSION_CONFIGURED = True


def _decompose_sdpa(graph_module):
    return lower_scaled_dot_product_attention(graph_module)


def _tensor_metadata(value):
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": str(value.device),
        "stride": list(value.stride()),
        "layout": str(value.layout),
    }


def _operator_names(graph_module):
    names = set()
    for node in graph_module.graph.nodes:
        if node.op not in ("call_function", "call_method", "call_module"):
            continue
        target = node.target
        if node.op == "call_function":
            target_name = ".".join(part for part in (
                getattr(target, "__module__", None),
                getattr(target, "__qualname__", getattr(target, "__name__", None)),
            ) if part)
        else:
            target_name = str(target)
        names.add(f"{node.op}:{target_name}")
    return sorted(names)


def _record_graph(
    model_name, component, graph_id, node_count, operators, lowered_operators,
    decomposed_sdpa, placement, error=None, phase=None, example_inputs=None, duration_ns=None,
    execution_index=None, compile_instance_id=None,
):
    from .work_metrics import record as record_metric
    if duration_ns is not None and phase:
        record_metric(phase, duration_ns / 1e9)
    record = {
        "model": model_name,
        "component": component,
        "graph_id": graph_id,
        "compile_instance_id": compile_instance_id,
        "nodes": node_count,
        "operators": operators,
        "lowered_operators": lowered_operators,
        "decomposed_sdpa": decomposed_sdpa,
        "placement": placement,
        "time_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        "time_monotonic_ns": time.monotonic_ns(),
    }
    if phase:
        record["phase"] = phase
    if example_inputs is not None:
        record["inputs"] = [_tensor_metadata(value) for value in example_inputs if isinstance(value, torch.Tensor)]
    if duration_ns is not None:
        record["duration_ns"] = duration_ns
    if execution_index is not None:
        record["execution_index"] = execution_index
    if error:
        record["compile_error"] = error.splitlines()[0][:500]
    audit_path = os.environ.get("FURIOSA_GRAPH_AUDIT_PATH")
    if audit_path:
        path = Path(audit_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(
        f"[Furiosa compiler] GRAPH AUDIT model={model_name} component={component} "
        f"graph={graph_id} nodes={node_count} placement={placement} "
        f"operators={len(operators)} lowered={len(lowered_operators)} "
        f"sdpa_decomposed={decomposed_sdpa}"
    )


@lru_cache(maxsize=None)
def _audited_backend(model_name: str, component: str):
    def compile_graph(graph_module, example_inputs):
        compile_instance_id = f"{os.getpid()}-{time.monotonic_ns()}"
        def record_graph(*args, **kwargs):
            kwargs.setdefault("compile_instance_id", compile_instance_id)
            _record_graph(*args, **kwargs)

        print(
            f"[Furiosa compiler] GRAPH CAPTURED model={model_name} component={component} "
            f"nodes={len(graph_module.graph.nodes)}",
            flush=True,
        )
        graph_text = str(graph_module.graph)
        graph_id = hashlib.sha256(graph_text.encode()).hexdigest()[:16]
        operators = _operator_names(graph_module)
        decomposed_sdpa = _decompose_sdpa(graph_module)
        lowered_operators = _operator_names(graph_module)
        record_graph(
            model_name, component, graph_id, len(graph_module.graph.nodes),
            operators, lowered_operators, decomposed_sdpa, "captured",
            phase="graph_captured", example_inputs=example_inputs,
        )
        compile_started_ns = time.monotonic_ns()
        record_graph(
            model_name, component, graph_id, len(graph_module.graph.nodes),
            operators, lowered_operators, decomposed_sdpa, "compile_started",
            phase="backend_compile_start", example_inputs=example_inputs,
        )
        try:
            print(
                f"[Furiosa compiler] NPU COMPILE START model={model_name} "
                f"component={component} nodes={len(graph_module.graph.nodes)}",
                flush=True,
            )
            with _record_function(f"compiler_compile::{model_name}::{component}::{graph_id}"):
                execute_npu = _STRICT_BACKEND(graph_module, example_inputs)
            print(
                f"[Furiosa compiler] NPU COMPILE READY model={model_name} "
                f"component={component}",
                flush=True,
            )
            compile_error = None
            record_graph(
                model_name, component, graph_id, len(graph_module.graph.nodes),
                operators, lowered_operators, decomposed_sdpa, "npu_compiled",
                phase="backend_returned", example_inputs=example_inputs,
                duration_ns=time.monotonic_ns() - compile_started_ns,
            )
        except Exception as exc:
            execute_npu = None
            compile_error = exc
            print(
                f"[Furiosa compiler] NPU COMPILE ERROR model={model_name} "
                f"component={component} error={str(exc).splitlines()[0][:500]}",
                flush=True,
            )
            if _require_npu_execution():
                record_graph(
                    model_name, component, graph_id, len(graph_module.graph.nodes),
                    operators, lowered_operators, decomposed_sdpa, "compile_failed", str(exc),
                    phase="backend_compile_failed", example_inputs=example_inputs,
                    duration_ns=time.monotonic_ns() - compile_started_ns,
                )
                raise RuntimeError(
                    f"NPU-only mode: strict compilation failed for {model_name}.{component}"
                ) from exc
        execute_npu_eager = None
        placement = None

        def cpu_fallback(args, error):
            nonlocal placement
            if _require_npu_execution():
                record_graph(
                    model_name, component, graph_id, len(graph_module.graph.nodes),
                    operators, lowered_operators, decomposed_sdpa, "cpu_fallback_blocked", str(error),
                    phase="cpu_fallback_blocked", example_inputs=example_inputs,
                )
                raise RuntimeError(
                    f"NPU-only mode: CPU fallback blocked for {model_name}.{component}"
                ) from error
            placement = "cpu"
            replace_graph_device(graph_module, "cpu")
            cpu_args = _move_tensors(args, torch.device("cpu"))
            record_graph(
                model_name, component, graph_id, len(graph_module.graph.nodes),
                operators, lowered_operators, decomposed_sdpa, placement, str(error),
                phase="cpu_fallback_started", example_inputs=example_inputs,
            )
            started_ns = time.monotonic_ns()
            with _record_function(f"cpu_fallback::{model_name}::{component}::{graph_id}"):
                with sdpa_kernel(SDPBackend.MATH):
                    result = _move_tensors(graph_module(*cpu_args), _NPU_DEVICE)
            record_graph(
                model_name, component, graph_id, len(graph_module.graph.nodes),
                operators, lowered_operators, decomposed_sdpa, placement, str(error),
                phase="cpu_fallback_completed", example_inputs=example_inputs,
                duration_ns=time.monotonic_ns() - started_ns,
            )
            return result

        def npu_eager_fallback(args, error):
            nonlocal placement, execute_npu_eager
            if execute_npu_eager is None:
                execute_npu_eager = _NPU_EAGER_BACKEND(graph_module, example_inputs)
            started_ns = time.monotonic_ns()
            with _record_function(f"npu_eager::{model_name}::{component}::{graph_id}"):
                result = execute_npu_eager(*args)
            placement = "npu_eager"
            record_graph(
                model_name, component, graph_id, len(graph_module.graph.nodes),
                operators, lowered_operators, decomposed_sdpa, placement, str(error),
                phase="npu_eager_completed", example_inputs=example_inputs,
                duration_ns=time.monotonic_ns() - started_ns,
            )
            return result

        execution_count = 0

        def run(*args):
            nonlocal placement, execution_count
            execution_count += 1
            execution_index = execution_count
            prepare_started_ns = time.monotonic_ns()
            record_graph(
                model_name, component, graph_id, len(graph_module.graph.nodes),
                operators, lowered_operators, decomposed_sdpa, "execution_preparing",
                phase="execution_prepare_start", example_inputs=args,
                execution_index=execution_index,
            )
            npu_args = _normalize_mixed_float_inputs(args)
            record_graph(
                model_name, component, graph_id, len(graph_module.graph.nodes),
                operators, lowered_operators, decomposed_sdpa, "execution_prepared",
                phase="execution_prepare_returned", example_inputs=npu_args,
                duration_ns=time.monotonic_ns() - prepare_started_ns,
                execution_index=execution_index,
            )
            if placement == "cpu":
                if _require_npu_execution():
                    raise RuntimeError(f"NPU-only mode: cached CPU placement blocked for {model_name}.{component}")
                cpu_args = _move_tensors(args, torch.device("cpu"))
                started_ns = time.monotonic_ns()
                with _record_function(f"cpu_fallback::{model_name}::{component}::{graph_id}"):
                    with sdpa_kernel(SDPBackend.MATH):
                        result = _move_tensors(graph_module(*cpu_args), _NPU_DEVICE)
                record_graph(
                    model_name, component, graph_id, len(graph_module.graph.nodes),
                    operators, lowered_operators, decomposed_sdpa, placement,
                    phase="cpu_repeat_completed", example_inputs=args,
                    duration_ns=time.monotonic_ns() - started_ns, execution_index=execution_index,
                )
                return result
            if placement == "npu_eager":
                with _record_function(f"npu_eager::{model_name}::{component}::{graph_id}"):
                    return execute_npu_eager(*npu_args)
            if compile_error is not None:
                try:
                    return npu_eager_fallback(npu_args, compile_error)
                except Exception as eager_exc:
                    return cpu_fallback(args, eager_exc)
            try:
                execute_started_ns = time.monotonic_ns()
                record_graph(
                    model_name, component, graph_id, len(graph_module.graph.nodes),
                    operators, lowered_operators, decomposed_sdpa, "npu_execution_started",
                    phase="first_execution_start" if execution_index == 1 else "repeat_execution_start",
                    example_inputs=npu_args, execution_index=execution_index,
                )
                print(
                    f"[Furiosa compiler] NPU EXECUTE START model={model_name} "
                    f"component={component}",
                    flush=True,
                )
                with _record_function(f"npu_graph::{model_name}::{component}::{graph_id}::compile-{compile_instance_id}::execution-{execution_index}"):
                    result = execute_npu(*npu_args)
                execute_duration_ns = time.monotonic_ns() - execute_started_ns
                print(
                    f"[Furiosa compiler] NPU EXECUTE READY model={model_name} "
                    f"component={component}",
                    flush=True,
                )
            except Exception as strict_exc:
                if _require_npu_execution():
                    record_graph(
                        model_name, component, graph_id, len(graph_module.graph.nodes),
                        operators, lowered_operators, decomposed_sdpa, "npu_execution_failed", str(strict_exc),
                        phase="npu_execution_failed", example_inputs=npu_args,
                        duration_ns=time.monotonic_ns() - execute_started_ns,
                        execution_index=execution_index,
                    )
                    raise RuntimeError(
                        f"NPU-only mode: execution failed for {model_name}.{component}"
                    ) from strict_exc
                try:
                    result = npu_eager_fallback(npu_args, strict_exc)
                except Exception as eager_exc:
                    return cpu_fallback(args, eager_exc)
            else:
                placement = "npu_compiled"
                record_graph(
                    model_name, component, graph_id, len(graph_module.graph.nodes),
                    operators, lowered_operators, decomposed_sdpa, placement,
                    phase="first_execution_completed" if execution_index == 1 else "repeat_execution_completed",
                    example_inputs=npu_args, duration_ns=execute_duration_ns,
                    execution_index=execution_index,
                )
            return result

        return run

    return compile_graph


class _CPUComponent(torch.nn.Module):
    """Keep an explicitly selected component on CPU and normalize its inputs."""

    def __init__(self, module):
        super().__init__()
        self.module = module

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("module"), name)

    def forward(self, *args, **kwargs):
        cpu = torch.device("cpu")
        with _record_function("cpu_model_component::explicit_cpu_component"):
            return self.module(*_move_tensors(args, cpu), **_move_tensors(kwargs, cpu))


class _NPUComponent(torch.nn.Module):
    def __init__(self, module, model_name, component):
        super().__init__()
        self.compiled = torch.compile(
            module,
            backend=_audited_backend(model_name, component),
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("compiled"), name)

    def forward(self, *args, **kwargs):
        npu_args = _move_tensors(args, _NPU_DEVICE)
        npu_kwargs = _move_tensors(kwargs, _NPU_DEVICE)
        with sdpa_kernel(SDPBackend.MATH):
            result = self.compiled(*npu_args, **npu_kwargs)
        return _move_tensors(result, torch.device("cpu"))


class _NPUBlock(torch.nn.Module):
    """Compile one NPU-resident block without moving its outputs back to CPU."""

    def __init__(self, module, model_name, component):
        super().__init__()
        self.component = component
        self._metadata_written = False
        conv = module
        self._operator_metadata = {
            "module_type": type(conv).__name__,
            "weight": _tensor_metadata(getattr(conv, "weight", None)),
            "bias": _tensor_metadata(getattr(conv, "bias", None)),
            "stride": list(getattr(conv, "stride", ())),
            "padding": list(getattr(conv, "padding", ())),
            "dilation": list(getattr(conv, "dilation", ())),
            "groups": getattr(conv, "groups", None),
        } if component == "unet.conv_in" else None
        self.compiled = torch.compile(
            module,
            backend=_audited_backend(model_name, component),
        )

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(super().__getattr__("compiled"), name)

    def forward(self, *args, **kwargs):
        npu = _NPU_DEVICE
        npu_args = _move_tensors(args, npu)
        npu_kwargs = _move_tensors(kwargs, npu)
        if self._operator_metadata is not None and not self._metadata_written:
            record = {
                "component": self.component,
                "inputs": [_tensor_metadata(value) for value in npu_args],
                "module": self._operator_metadata,
            }
            audit_path = os.environ.get("FURIOSA_OPERATOR_AUDIT_PATH")
            if audit_path:
                path = Path(audit_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a") as stream:
                    stream.write(json.dumps(record) + "\n")
            print(f"[Furiosa compiler] OPERATOR METADATA {json.dumps(record)}", flush=True)
            self._metadata_written = True
        result = self.compiled(*npu_args, **npu_kwargs)
        if os.environ.get("FURIOSA_STOP_AFTER_COMPONENT") == self.component:
            raise RuntimeError(f"diagnostic stop after {self.component}")
        return result


class _SDXLConvInSpatialLinear(torch.nn.Module):
    """Split SDXL's 3x3 conv_in into nine small, independently compiled linears."""

    def __init__(self, conv, model_name):
        super().__init__()
        if (conv.in_channels, conv.out_channels, tuple(conv.kernel_size), tuple(conv.stride),
                tuple(conv.padding), tuple(conv.dilation), conv.groups, conv.padding_mode) != (
                4, 320, (3, 3), (1, 1), (1, 1), (1, 1), 1, "zeros"):
            raise NotImplementedError("spatial-linear route only supports the SDXL Turbo conv_in shape")
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_h, self.kernel_w = conv.kernel_size
        self.stride_h, self.stride_w = conv.stride
        self.pad_h, self.pad_w = conv.padding
        self.taps = torch.nn.ModuleList()
        for kh in range(self.kernel_h):
            for kw in range(self.kernel_w):
                has_bias = conv.bias is not None and kh == 0 and kw == 0
                linear = torch.nn.Linear(
                    self.in_channels, self.out_channels, bias=has_bias,
                    device=conv.weight.device, dtype=conv.weight.dtype,
                )
                with torch.no_grad():
                    linear.weight.copy_(conv.weight[:, :, kh, kw])
                    if has_bias:
                        linear.bias.copy_(conv.bias)
                # Reuse one backend identity so Dynamo does not hit its
                # recompile_limit on the ninth equivalent Linear module.
                self.taps.append(_NPUBlock(linear, model_name, "unet.conv_in.spatial_tap"))

    def forward(self, value):
        if value.ndim != 4 or value.shape[1] != self.in_channels or value.dtype != torch.bfloat16:
            raise ValueError("SDXL conv_in spatial-linear route expects BF16 NCHW with four channels")
        from .tcl_windows import pack_windows
        with _record_function("transfer::conv_window_input_to_npu"):
            value = value.to("furiosa:0")
        packed = pack_windows(value, self.kernel_h, self.pad_h, self.stride_h)
        windows = [packed[:, :, :, kh, kw, :].contiguous()
                   for kh in range(self.kernel_h) for kw in range(self.kernel_w)]
        output = None
        for tap, window in zip(self.taps, windows):
            # Each tap is rounded to BF16 by the NPU linear kernel. Accumulate
            # those partial sums in FP32 on the NPU to avoid repeated BF16
            # rounding across the 3x3 kernel.
            contribution = tap(window).float()
            output = contribution if output is None else output + contribution
        return output.permute(0, 3, 1, 2).contiguous().to(torch.bfloat16)


class _LinearFromWeight(torch.nn.Module):
    def forward(self, value, weight, bias=None):
        return F.linear(value, weight, bias)


class _NPUConv2dSpatialTaps(torch.nn.Module):
    """Decompose BF16 Conv2d to one NPU Linear over concatenated spatial taps."""

    def __init__(self, conv, model_name, component):
        super().__init__()
        if conv.groups != 1 or conv.dilation != (1, 1):
            raise NotImplementedError("spatial-tap Conv2d supports groups=1 and dilation=1")
        if any(pair[0] != pair[1] for pair in (conv.kernel_size, conv.padding, conv.stride)):
            raise NotImplementedError("TCL window packing currently requires symmetric kernel, padding and stride")
        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.kernel_h, self.kernel_w = conv.kernel_size
        self.stride_h, self.stride_w = conv.stride
        self.pad_h, self.pad_w = conv.padding
        self.output_dtype = conv.weight.dtype
        # One block matrix exactly represents the sum of every spatial tap.
        # Keep construction on CPU, then transfer the combined weight once.
        weight = conv.weight.detach().to("cpu").permute(0, 2, 3, 1).contiguous()
        weight = weight.reshape(self.out_channels, self.kernel_h * self.kernel_w * self.in_channels)
        self.combined_weight = torch.nn.Parameter(
            weight.to("furiosa:0"), requires_grad=False
        )
        self.bias = None if conv.bias is None else torch.nn.Parameter(
            conv.bias.detach().to(device="furiosa:0", dtype=self.output_dtype),
            requires_grad=False,
        )
        self.linear = _NPUBlock(
            _LinearFromWeight(), model_name,
            f"unet.spatial_conv.{self.in_channels}x{self.out_channels}x"
            f"{self.kernel_h}x{self.kernel_w}",
        )

    def forward(self, value):
        if value.ndim != 4 or value.shape[1] != self.in_channels or value.dtype != torch.bfloat16:
            raise ValueError("spatial-tap Conv2d expects BF16 NCHW input with matching channels")
        from .tcl_windows import pack_windows
        height, width = value.shape[-2:]
        out_h = (height + 2 * self.pad_h - self.kernel_h) // self.stride_h + 1
        out_w = (width + 2 * self.pad_w - self.kernel_w) // self.stride_w + 1
        with _record_function("transfer::conv_window_input_to_npu"):
            value = value.to("furiosa:0")
        packed = pack_windows(value, self.kernel_h, self.pad_h, self.stride_h)
        joined = packed.reshape(-1, self.kernel_h * self.kernel_w * self.in_channels)
        output = self.linear(joined, self.combined_weight, self.bias)
        output = output.reshape(value.shape[0], out_h, out_w, self.out_channels)
        return output.permute(0, 3, 1, 2).contiguous().to(self.output_dtype)


def _segment_unet_for_npu(module, model_name):
    count = 0
    for name in (
        "conv_in", "time_proj", "time_embedding", "add_embedding",
        "encoder_hid_proj", "conv_norm_out", "conv_act", "conv_out",
    ):
        child = getattr(module, name, None)
        if isinstance(child, torch.nn.Module):
            if name == "conv_in" and isinstance(child, torch.nn.Conv2d):
                try:
                    child = _SDXLConvInSpatialLinear(child, model_name)
                    if os.environ.get("FURIOSA_NPU_AUDIT_DIR"):
                        from .execution_auditor import instrument_callable
                        instrument_callable(child, "forward", "model_component::unet.conv_in")
                    print("[Furiosa backend] unet.conv_in uses nine strict NPU spatial-linear graphs", flush=True)
                except NotImplementedError:
                    child = _NPUBlock(child, model_name, f"unet.{name}")
            else:
                child = _NPUBlock(child, model_name, f"unet.{name}")
            setattr(module, name, child)
            count += 1
    for name in ("down_blocks", "mid_block", "up_blocks"):
        child = getattr(module, name, None)
        if isinstance(child, torch.nn.ModuleList):
            for index, block in enumerate(child):
                operator_blocks = {
                    part.strip() for part in os.environ.get(
                        "FURIOSA_SEGMENT_UNET_OPERATOR_BLOCKS", "0"
                    ).split(",") if part.strip()
                }
                if (
                    name in ("down_blocks", "up_blocks")
                    and os.environ.get("FURIOSA_SEGMENT_UNET_OPERATORS") == "1"
                    and str(index) in operator_blocks
                ):
                    prefix = f"unet.{name}.{index}"
                    count += _segment_unet_operator_block(block, model_name, prefix)
                    if os.environ.get("FURIOSA_NPU_AUDIT_DIR"):
                        from .execution_auditor import instrument_callable
                        instrument_callable(block, "forward", f"model_component::{prefix}")
                else:
                    child[index] = _NPUBlock(block, model_name, f"unet.{name}.{index}")
                    count += 1
        elif isinstance(child, torch.nn.Module):
            if name == "mid_block" and os.environ.get("FURIOSA_SEGMENT_UNET_OPERATORS") == "1":
                count += _segment_unet_operator_block(child, model_name, "unet.mid_block")
                if os.environ.get("FURIOSA_NPU_AUDIT_DIR"):
                    from .execution_auditor import instrument_callable
                    instrument_callable(child, "forward", "model_component::unet.mid_block")
            else:
                setattr(module, name, _NPUBlock(child, model_name, f"unet.{name}"))
                count += 1
    return count


def _segment_down_block_operators(block, model_name, block_index):
    return _segment_unet_operator_block(block, model_name, f"unet.down_blocks.{block_index}")


def _segment_unet_operator_block(block, model_name, component_prefix):
    recompile_limit = int(os.environ.get("FURIOSA_DYNAMO_RECOMPILE_LIMIT", "512"))
    if torch._dynamo.config.recompile_limit < recompile_limit:
        torch._dynamo.config.recompile_limit = recompile_limit
    count = 0
    resnets = getattr(block, "resnets", None)
    if isinstance(resnets, torch.nn.ModuleList):
        for resnet in resnets:
            for name in ("norm1", "nonlinearity", "conv1", "time_emb_proj",
                         "norm2", "conv2", "conv_shortcut"):
                child = getattr(resnet, name, None)
                if not isinstance(child, torch.nn.Module) or isinstance(child, _NPUBlock):
                    continue
                kind = {
                    "norm1": "norm", "norm2": "norm",
                    "conv1": "conv", "conv2": "conv", "conv_shortcut": "conv",
                    "time_emb_proj": "time_linear", "nonlinearity": "activation",
                }[name]
                spatial_signature = os.environ.get("FURIOSA_SPATIAL_CONV_SIGNATURES", "")
                conv_signature = (
                    f"{getattr(child, 'in_channels', 0)}x{getattr(child, 'out_channels', 0)}x"
                    f"{getattr(child, 'kernel_size', (0, 0))[0]}x"
                    f"{getattr(child, 'kernel_size', (0, 0))[1]}"
                )
                if kind == "conv" and conv_signature in spatial_signature.split(","):
                    child = _NPUConv2dSpatialTaps(
                        child, model_name,
                        f"unet.spatial_conv.{conv_signature}",
                    )
                else:
                    child = _NPUBlock(
                        child, model_name, f"{component_prefix}.resnet.{kind}"
                    )
                setattr(resnet, name, child)
                count += 1
    downsamplers = getattr(block, "downsamplers", None)
    if isinstance(downsamplers, torch.nn.ModuleList):
        for downsampler in downsamplers:
            conv = getattr(downsampler, "conv", None)
            if isinstance(conv, torch.nn.Module) and not isinstance(conv, _NPUBlock):
                downsampler.conv = _NPUBlock(
                    conv, model_name, f"{component_prefix}.downsample.conv"
                )
                count += 1
    upsamplers = getattr(block, "upsamplers", None)
    if isinstance(upsamplers, torch.nn.ModuleList):
        for upsampler in upsamplers:
            conv = getattr(upsampler, "conv", None)
            if isinstance(conv, torch.nn.Module) and not isinstance(conv, _NPUBlock):
                component = f"{component_prefix}.upsample.conv"
                upsampler.conv = _segment_attention_stage(conv, model_name, component)
                count += 1
    attentions = getattr(block, "attentions", None)
    if isinstance(attentions, torch.nn.ModuleList):
        for attention_index, attention in enumerate(attentions):
            for name in ("norm", "proj_in"):
                child = getattr(attention, name, None)
                if isinstance(child, torch.nn.Module) and not isinstance(child, _NPUBlock):
                    component = f"{component_prefix}.attention{attention_index}.{name}"
                    setattr(attention, name, _segment_attention_stage(
                        child, model_name, component
                    ))
                    count += 1
            transformer_blocks = getattr(attention, "transformer_blocks", None)
            if isinstance(transformer_blocks, torch.nn.ModuleList):
                for transformer_index, child in enumerate(transformer_blocks):
                    if isinstance(child, torch.nn.Module) and not isinstance(child, _NPUBlock):
                        for child_name in ("norm1", "attn1", "norm2", "attn2", "norm3", "ff"):
                            submodule = getattr(child, child_name, None)
                            if not isinstance(submodule, torch.nn.Module) or isinstance(submodule, _NPUBlock):
                                continue
                            component = (
                                f"{component_prefix}.attention{attention_index}"
                                f".block{transformer_index}"
                                f".{child_name}"
                            )
                            if child_name in ("attn1", "attn2"):
                                count += _segment_attention_projections(
                                    submodule, model_name, component
                                )
                                continue
                            setattr(child, child_name, _NPUBlock(
                                submodule, model_name, component,
                            ))
                            count += 1
            child = getattr(attention, "proj_out", None)
            if isinstance(child, torch.nn.Module) and not isinstance(child, _NPUBlock):
                attention.proj_out = _segment_attention_stage(
                    child, model_name,
                    f"{component_prefix}.attention{attention_index}.proj_out",
                )
                count += 1
    return count


def _segment_attention_stage(module, model_name, component):
    if isinstance(module, torch.nn.Conv2d):
        signature = (
            f"{module.in_channels}x{module.out_channels}x"
            f"{module.kernel_size[0]}x{module.kernel_size[1]}"
        )
        if signature in os.environ.get("FURIOSA_SPATIAL_CONV_SIGNATURES", "").split(","):
            return _NPUConv2dSpatialTaps(module, model_name, component)
    return _NPUBlock(module, model_name, component)


def _segment_attention_projections(attention, model_name, component):
    """Compile attention projections separately; leave public SDPA math in the caller."""
    count = 0
    original_forward = attention.forward

    def forward_with_manual_sdpa(*args, **kwargs):
        original_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = manual_scaled_dot_product_attention
        try:
            with _record_function(f"npu_eager::{component}.sdpa"):
                return original_forward(*args, **kwargs)
        finally:
            F.scaled_dot_product_attention = original_sdpa

    attention.forward = forward_with_manual_sdpa
    for name in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out"):
        child = getattr(attention, name, None)
        if isinstance(child, torch.nn.Module) and not isinstance(child, _NPUBlock):
            setattr(attention, name, _NPUBlock(child, model_name, f"{component}.{name}"))
            count += 1
    to_out = getattr(attention, "to_out", None)
    if isinstance(to_out, (torch.nn.ModuleList, torch.nn.Sequential)) and len(to_out):
        child = to_out[0]
        if isinstance(child, torch.nn.Module) and not isinstance(child, _NPUBlock):
            to_out[0] = _NPUBlock(child, model_name, f"{component}.to_out")
            count += 1
    for name in ("group_norm", "spatial_norm"):
        child = getattr(attention, name, None)
        if isinstance(child, torch.nn.Module) and not isinstance(child, _NPUBlock):
            setattr(attention, name, _NPUBlock(child, model_name, f"{component}.{name}"))
            count += 1
    return count


def _segment_clip_text_encoder(module, model_name, component):
    """Compile public CLIP text-model stages as separate NPU graphs."""
    count = 0
    text_model = getattr(module, "text_model", None)
    if not isinstance(text_model, torch.nn.Module):
        text_model = module
    child = getattr(text_model, "embeddings", None)
    if isinstance(child, torch.nn.Module):
        text_model.embeddings = _NPUBlock(child, model_name, f"{component}.embeddings")
        count += 1
    encoder = getattr(text_model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if isinstance(layers, torch.nn.ModuleList):
        for index, layer in enumerate(layers):
            layers[index] = _NPUBlock(layer, model_name, f"{component}.encoder.layers.{index}")
            count += 1
    child = getattr(text_model, "final_layer_norm", None)
    if isinstance(child, torch.nn.Module):
        text_model.final_layer_norm = _NPUBlock(child, model_name, f"{component}.final_layer_norm")
        count += 1
    child = getattr(module, "text_projection", None)
    if isinstance(child, torch.nn.Module):
        module.text_projection = _NPUBlock(child, model_name, f"{component}.text_projection")
        count += 1
    return count


def keep_scheduler_latents_on_cpu(pipe):
    """Keep scheduler-owned scalar tensors and sample state on their CPU device."""
    prepare_latents = pipe.prepare_latents

    def prepare(*args, **kwargs):
        args = list(args)
        if len(args) > 5:
            args[5] = torch.device("cpu")
            kwargs.pop("device", None)
        else:
            kwargs["device"] = torch.device("cpu")
        return prepare_latents(*args, **kwargs)

    pipe.prepare_latents = prepare


def install_npu_scheduler_sigma_scale(pipe, model_name):
    """Generate SDXL latents as before, then scale them in a strict NPU graph."""
    from diffusers.utils.torch_utils import randn_tensor

    sigma = pipe.scheduler.init_noise_sigma
    if isinstance(sigma, torch.Tensor):
        if sigma.numel() != 1:
            raise ValueError(f"Expected scalar scheduler init_noise_sigma, got shape {tuple(sigma.shape)}")
        sigma_value = float(sigma.detach().cpu().item())
        sigma_source = f"tensor:{sigma.device}:{sigma.dtype}:{tuple(sigma.shape)}"
    else:
        sigma_value = float(sigma)
        sigma_source = type(sigma).__name__

    class _ScaleLatents(torch.nn.Module):
        def forward(self, latents):
            return latents * sigma_value

    scale_latents = _NPUBlock(_ScaleLatents(), model_name, "scheduler.init_noise_sigma")

    def prepare_latents(batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // pipe.vae_scale_factor,
            int(width) // pipe.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)
        return scale_latents(latents)

    pipe.prepare_latents = prepare_latents
    print(
        f"[Furiosa backend] {model_name}.scheduler.init_noise_sigma={sigma_value} "
        f"({sigma_source}); scaling latents through a strict NPU graph",
        flush=True,
    )


def install_cpu_scheduler_execution(pipe):
    """Keep Diffusers scheduler state and arithmetic on CPU between NPU model calls."""
    scheduler = pipe.scheduler
    set_timesteps = scheduler.set_timesteps
    scale_model_input = scheduler.scale_model_input
    step = scheduler.step

    def set_cpu_timesteps(*args, **kwargs):
        args = list(args)
        if len(args) > 1:
            args[1] = torch.device("cpu")
        else:
            kwargs["device"] = torch.device("cpu")
        return set_timesteps(*args, **kwargs)

    def scale_on_cpu(sample, timestep, *args, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to("cpu")
        return scale_model_input(sample.to("cpu"), timestep, *args, **kwargs)

    def step_on_cpu(model_output, timestep, sample, *args, **kwargs):
        if isinstance(timestep, torch.Tensor):
            timestep = timestep.to("cpu")
        return step(model_output.to("cpu"), timestep, sample.to("cpu"), *args, **kwargs)

    scheduler.set_timesteps = set_cpu_timesteps
    scheduler.scale_model_input = scale_on_cpu
    scheduler.step = step_on_cpu
    print("[Furiosa backend] Diffusers scheduler state and math remain explicitly on CPU", flush=True)


def enable_npu_graphs(pipe, model_name: str) -> tuple[str, ...]:
    _configure_fused_pes()
    install_device_safe_sdpa()
    install_dtype_promoting_cat()
    if _require_npu_execution():
        install_group_norm_decomposition()
        print("[Furiosa backend] strict mode uses public mean-square GroupNorm decomposition", flush=True)
    if "qwen" in model_name.lower():
        install_qwen_rotary_device_fix()
    enabled = []
    cpu_components = {
        name.strip()
        for name in os.environ.get("FURIOSA_CPU_COMPONENTS", "").split(",")
        if name.strip()
    }
    fusion = get_fusion()
    print(
        f"[Furiosa compiler] Rayon threads={os.environ['RAYON_NUM_THREADS']}, "
        f"OMP threads={os.environ['OMP_NUM_THREADS']}; vision tactic enabled; "
        f"device={_NPU_DEVICE}; fusion={fusion.num_fusion} PEs; mesh=Single"
    )
    if _require_npu_execution() and cpu_components:
        raise RuntimeError(f"NPU-only mode forbids CPU components: {sorted(cpu_components)}")
    for name in _COMPONENTS:
        module = getattr(pipe, name, None)
        if isinstance(module, torch.nn.Module):
            if name in cpu_components:
                module.to("cpu")
                setattr(pipe, name, _CPUComponent(module))
                enabled.append(name)
                print(
                    f"[Furiosa backend] {model_name}.{name}: explicitly kept on CPU "
                    f"(FURIOSA_CPU_COMPONENTS)",
                    flush=True,
                )
                continue
            try:
                module.to(_NPU_DEVICE)
            except RuntimeError as exc:
                if _require_npu_execution():
                    raise RuntimeError(
                        f"NPU-only mode: failed to place {model_name}.{name} on {_NPU_DEVICE}"
                    ) from exc
                module.to("cpu")
                print(f"[Furiosa backend] {model_name}.{name}: left on CPU ({str(exc).splitlines()[0]})")
                continue
            if name in ("text_encoder", "text_encoder_2") and os.environ.get("FURIOSA_SEGMENT_TEXT_ENCODERS") == "1":
                count = _segment_clip_text_encoder(module, model_name, name)
                setattr(pipe, name, module)
                if os.environ.get("FURIOSA_NPU_AUDIT_DIR"):
                    from .execution_auditor import instrument_callable
                    instrument_callable(module, "forward", f"model_component::{name}")
                enabled.append(name)
                print(f"[Furiosa backend] {model_name}.{name}: segmented into {count} NPU-compiled CLIP stages", flush=True)
            elif name == "unet" and os.environ.get("FURIOSA_SEGMENT_UNET") == "1":
                count = _segment_unet_for_npu(module, model_name)
                setattr(pipe, name, module)
                if os.environ.get("FURIOSA_NPU_AUDIT_DIR"):
                    from .execution_auditor import instrument_callable
                    instrument_callable(module, "forward", f"model_component::{name}")
                enabled.append(name)
                print(
                    f"[Furiosa backend] {model_name}.{name}: segmented into {count} "
                    "NPU-compiled submodules",
                    flush=True,
                )
            else:
                component_wrapper = _NPUComponent(module, model_name, name)
                if os.environ.get("FURIOSA_NPU_AUDIT_DIR"):
                    from .execution_auditor import instrument_callable
                    instrument_callable(component_wrapper, "forward", f"model_component::{name}")
                setattr(pipe, name, component_wrapper)
                enabled.append(name)
                print(f"[Furiosa backend] {model_name}.{name}: lazy NPU graph compilation enabled")

    if not enabled:
        raise RuntimeError(f"No compilable Diffusers components found for {model_name}")
    if _require_npu_execution():
        print("[Furiosa backend] NPU-only strict mode enabled; CPU and eager fallbacks are disabled")
    else:
        print("[Furiosa backend] Unsupported graphs are audited and may use fallback")
    return tuple(enabled)
