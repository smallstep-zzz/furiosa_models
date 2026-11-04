"""Adapters for installed public TCL examples with explicit specialization limits."""
from __future__ import annotations
from functools import lru_cache

import torch


def _supports_conv2d_stride2_demo(value, filt, bias):
    return (
        isinstance(value, torch.Tensor)
        and isinstance(filt, torch.Tensor)
        and isinstance(bias, torch.Tensor)
        and tuple(value.shape) == (2, 3, 8, 8)
        and tuple(filt.shape) == (5, 3, 3, 3)
        and tuple(bias.shape) == (5,)
        and value.dtype == filt.dtype == bias.dtype == torch.bfloat16
        and value.device.type == filt.device.type == bias.device.type == "furiosa"
    )


@lru_cache(maxsize=1)
def _compiled_conv2d_stride2_demo():
    from furiosa.tcl.examples.convolution import build_conv2d_stride2_kernel
    from furiosa.torch import CompileModule, TclModule

    module = TclModule(build_conv2d_stride2_kernel())
    example = (
        torch.randn(2, 3, 8, 8, dtype=torch.bfloat16),
        torch.randn(5, 3, 3, 3, dtype=torch.bfloat16),
        torch.randn(5, dtype=torch.bfloat16),
    )
    exported = torch.export.export(module, example, strict=False)
    return CompileModule.from_exported(exported).to(torch.device("furiosa:0"))


def conv2d_stride2_demo(value, filt, bias):
    """Execute the SDK's fixed [2,3,8,8] BF16 stride-2 conv2d sample on RNGD."""
    if not _supports_conv2d_stride2_demo(value, filt, bias):
        raise ValueError("TCL demo accepts only BF16 furiosa tensors [2,3,8,8], [5,3,3,3], [5]")
    return _compiled_conv2d_stride2_demo()(value, filt, bias)[0]


from .operator_compat import register_operator

register_operator(
    "conv2d",
    "tcl",
    conv2d_stride2_demo,
    supports=_supports_conv2d_stride2_demo,
)



def _supports_sdxl_conv_in(value, filt, bias):
    return (
        isinstance(value, torch.Tensor)
        and isinstance(filt, torch.Tensor)
        and isinstance(bias, torch.Tensor)
        and tuple(value.shape) == (1, 4, 64, 64)
        and tuple(filt.shape) == (320, 4, 3, 3)
        and tuple(bias.shape) == (320,)
        and value.dtype == filt.dtype == bias.dtype == torch.bfloat16
        and value.device.type == filt.device.type == bias.device.type == "furiosa"
    )


@lru_cache(maxsize=1)
def _compiled_sdxl_conv_in():
    import furiosa.tcl as tcl
    from furiosa.tcl.examples.convolution import conv2d_kernel
    from furiosa.torch import CompileModule, TclModule

    bindings = {
        "N": 1, "C": 4, "H": 64, "W": 64, "K": 8,
        "Hw": 3, "Ww": 3, "Hf": 64, "Wf": 64,
    }
    kernel = tcl.specialize_kernel(
        conv2d_kernel, bindings,
        constexpr={"padding": 1, "stride": 1, "dilation": 1},
    )
    module = TclModule(kernel)
    example = (
        torch.randn(1, 4, 64, 64, dtype=torch.bfloat16),
        torch.randn(8, 4, 3, 3, dtype=torch.bfloat16),
        torch.randn(8, dtype=torch.bfloat16),
    )
    exported = torch.export.export(module, example, strict=False)
    return CompileModule.from_exported(exported).to(torch.device("furiosa:0"))


def sdxl_conv_in(value, filt, bias):
    # Fixed-signature wrapper; each tile is compiled and run on RNGD.
    if not _supports_sdxl_conv_in(value, filt, bias):
        raise ValueError("SDXL conv_in TCL accepts BF16 furiosa [1,4,64,64], [320,4,3,3], [320]")
    compiled = _compiled_sdxl_conv_in()
    output_tiles = [
        compiled(value, filt[start:start + 8], bias[start:start + 8])[0]
        for start in range(0, 320, 8)
    ]
    return torch.cat(output_tiles, dim=1)


register_operator(
    "conv2d",
    "tcl",
    sdxl_conv_in,
    supports=_supports_sdxl_conv_in,
)
