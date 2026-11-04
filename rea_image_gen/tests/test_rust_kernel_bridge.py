import unittest
import torch

from compiler_engine.rust_kernel_bridge import rust_add_one_i32


class RustKernelBridgeTests(unittest.TestCase):
    def test_rejects_wrong_dtype_before_loading_extension(self):
        with self.assertRaisesRegex(ValueError, "torch.int32"):
            rust_add_one_i32(torch.zeros(2048, dtype=torch.float32))

    def test_rejects_wrong_shape_before_loading_extension(self):
        with self.assertRaisesRegex(ValueError, "2048"):
            rust_add_one_i32(torch.zeros(8, dtype=torch.int32))


if __name__ == "__main__":
    unittest.main()
