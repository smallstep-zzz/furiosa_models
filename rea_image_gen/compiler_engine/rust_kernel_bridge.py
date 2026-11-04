"""Opt-in Python bridge to public Furiosa vISA kernels."""

import ctypes
import os
from functools import lru_cache
from pathlib import Path

import torch


_DEFAULT_LIBRARY = (
    Path(__file__).resolve().parent.parent
    / "rust_kernels"
    / "target"
    / "release"
    / "librngd_kernels.so"
)
_COUNT = 2048
_I32_PTR = ctypes.POINTER(ctypes.c_int32)


@lru_cache(maxsize=4)
def _load_library(path: str):
    library = ctypes.CDLL(path)
    function = library.rngd_add_one_i32
    function.argtypes = (_I32_PTR, _I32_PTR, ctypes.c_size_t)
    function.restype = ctypes.c_int
    return function


def rust_add_one_i32(value: torch.Tensor) -> torch.Tensor:
    """Run the fixed-shape Rust NPU add-one proof kernel via host staging."""
    if not isinstance(value, torch.Tensor) or value.dtype != torch.int32 or value.numel() != _COUNT:
        raise ValueError("expected a torch.int32 tensor with exactly 2048 values")
    original_device = value.device
    source = value.detach().contiguous().reshape(-1).cpu()
    destination = torch.empty_like(source)
    library_path = os.environ.get("FURIOSA_RUST_KERNEL_LIB", str(_DEFAULT_LIBRARY))
    function = _load_library(library_path)
    status = function(
        ctypes.cast(source.data_ptr(), _I32_PTR),
        ctypes.cast(destination.data_ptr(), _I32_PTR),
        _COUNT,
    )
    if status:
        raise RuntimeError(f"Rust vISA kernel failed with status {status}")
    return destination.reshape(value.shape).to(original_device)


from .operator_compat import register_operator

register_operator("rngd.add_one_i32", "rust_visa", rust_add_one_i32)
