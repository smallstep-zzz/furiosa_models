"""FP32 resident YOLO adapter using shared native SDK contraction kernels."""
from functools import lru_cache
import hashlib
import json
import os
import time
import torch
import torch.nn.functional as F
from furiosa.torch import CompileModule
from furiosa.torch.config import CompilerConfig
from compiler_engine.renegade_diffusion_compiler import _record_graph
from compiler_engine.execution_auditor import record_function
from conv_adapter import SpatialLinearConv2d
from pool_adapter import ResidentMaxPool

CONFIG = CompilerConfig(implicit_type_casting=False, implicit_type_casting_all_activation=False)


class Linear(torch.nn.Module):
    def forward(self, value, weight, bias):
        return F.linear(value, weight, bias)


@lru_cache(maxsize=128)
def build_linear(shapes):
    gid = hashlib.sha256(json.dumps(['linear', shapes, 'fp32']).encode()).hexdigest()[:16]
    instance = f'{os.getpid()}-{time.monotonic_ns()}'
    print('compile_linear '+str(shapes), flush=True)
    args = tuple(torch.empty(shape, dtype=torch.float32) for shape in shapes)
    ep = torch.export.export(Linear(), args)
    operators = [str(n.target) for n in ep.graph.nodes if n.op == 'call_function']
    with record_function('compiler_compile::yolo11::'+gid):
        exe = CompileModule.from_exported(ep, compiler_config=CONFIG).to('furiosa:0')
    return exe, gid, instance, operators


COUNTS = {}


class ResidentConv(SpatialLinearConv2d):
    def pack(self, value):
        from fast_pack import compiled_pack
        return compiled_pack(self, value)

    def __init__(self, conv, name):
        super().__init__(conv)
        self.name = name
        if self.linear.bias is None:
            self.linear.bias = torch.nn.Parameter(torch.zeros(self.out_channels), requires_grad=False)
        padded_out = max(16, 1 << (self.out_channels-1).bit_length())
        if padded_out != self.out_channels:
            self.linear.weight = torch.nn.Parameter(F.pad(self.linear.weight, (0, 0, 0, padded_out-self.out_channels)), requires_grad=False)
            self.linear.bias = torch.nn.Parameter(F.pad(self.linear.bias, (0, padded_out-self.out_channels)), requires_grad=False)

    def forward(self, value):
        assert value.device.type == 'furiosa'
        with record_function('model_component::'+self.name+'::pack'):
            packed = self.pack(value)
            rows = packed.shape[0]
            if rows % 32:
                packed = F.pad(packed, (0, 0, 0, (-rows) % 32))
            args = (packed, self.linear.weight, self.linear.bias)
            assert all(t.device.type == 'furiosa' for t in args)
        shapes = tuple(tuple(t.shape) for t in args)
        exe, gid, instance, operators = build_linear(shapes)
        key = (self.name, gid, instance)
        index = COUNTS.get(key, 0)+1
        COUNTS[key] = index
        if index == 1:
            _record_graph('yolo11', self.name, gid, 1, operators, operators, False,
                          'npu_compiled', phase='backend_returned', compile_instance_id=instance)
        def event(phase, placement):
            if index == 1 or os.environ.get('FURIOSA_TORCH_PROFILE') == 'trace':
                _record_graph('yolo11', self.name, gid, 1, operators, operators, False,
                              placement, phase=phase, example_inputs=args, execution_index=index,
                              compile_instance_id=instance)
        event('first_execution_start' if index == 1 else 'repeat_execution_start', 'npu_execution_started')
        with record_function(f'npu_graph::yolo11::{self.name}::{gid}::compile-{instance}::execution-{index}'):
            result = exe(*args)
        event('first_execution_completed' if index == 1 else 'repeat_execution_completed', 'npu_compiled')
        return self.unpack(result[:rows, :self.out_channels], value.shape)


def adapt(model):
    from fused_model import FusedConv
    from fused_pool import ResidentMaxPool

    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Conv2d):
            eligible = module.kernel_size == (3,3) and module.in_channels != 3
            model.set_submodule(name, FusedConv(module,name) if eligible else ResidentConv(module, name))
        elif isinstance(module, torch.nn.MaxPool2d):
            model.set_submodule(name, ResidentMaxPool(module))
    return model
