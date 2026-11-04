"""Compose public SiLU with the unchanged compensated convolution bridge."""
from functools import lru_cache
import torch
from furiosa.torch import CompileModule
from furiosa.torch.config import CompilerConfig
from tcl_conv_candidate import CompensatedConv


class CompensatedSiLU(CompensatedConv):
    def forward(self, value, weight_hi, weight_lo, bias):
        result = super().forward(value, weight_hi, weight_lo, bias)[0]
        return (result * torch.sigmoid(result),)


@lru_cache(maxsize=64)
def build(shape, weight_shape, padding=1, stride=2):
    args = (torch.empty(shape), torch.empty(weight_shape),
            torch.empty(weight_shape), torch.empty(weight_shape[0]))
    ep = torch.export.export(
        CompensatedSiLU(shape, weight_shape, padding, stride), args, strict=False)
    config = CompilerConfig(implicit_type_casting=False,
                            implicit_type_casting_all_activation=False)
    return CompileModule.from_exported(ep, compiler_config=config).to('furiosa:0')
