import unittest

from compiler_engine.operator_compat import register_operator, run_operator


class OperatorCompatTests(unittest.TestCase):
    def test_selects_first_supported_route_by_public_fallback_order(self):
        calls = []
        register_operator("test.route.order", "rust_visa", lambda x: calls.append("rust") or x + 3)
        register_operator("test.route.order", "tcl", lambda x: calls.append("tcl") or x + 2)
        register_operator("test.route.order", "python_decomposition", lambda x: calls.append("python") or x + 1)
        self.assertEqual(run_operator("test.route.order", 4), 5)
        self.assertEqual(calls, ["python"])

    def test_skips_routes_with_rejected_shape_guard(self):
        register_operator("test.route.guard", "tcl", lambda x: x + 2, supports=lambda x: x == 8)
        register_operator("test.route.guard", "rust_visa", lambda x: x + 3)
        self.assertEqual(run_operator("test.route.guard", 4), 7)

    def test_reports_when_no_registered_route_accepts_inputs(self):
        register_operator("test.route.none", "tcl", lambda x: x, supports=lambda x: False)
        with self.assertRaisesRegex(LookupError, "no compatible route"):
            run_operator("test.route.none", 4)


if __name__ == "__main__":
    unittest.main()
