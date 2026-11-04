#!/usr/bin/env python3
"""Separate-process RNGD smoke for the Rust vISA Python host-staging bridge."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from compiler_engine.operator_compat import run_operator
from compiler_engine.rust_kernel_bridge import rust_add_one_i32

source = torch.arange(2048, dtype=torch.int32)
actual = run_operator("rngd.add_one_i32", source)
torch.testing.assert_close(actual, source + 1)
print(f"OK rust_visa n=2048 device=RNGD result_device={actual.device.type} exact=true")
