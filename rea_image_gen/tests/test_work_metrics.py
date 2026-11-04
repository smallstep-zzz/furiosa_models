import unittest
from compiler_engine.work_metrics import snapshot,record,difference
class MetricsTests(unittest.TestCase):
    def test_request_delta_excludes_previous_request(self):
        record('test_compile',7)
        before=snapshot()
        record('test_compile',2)
        record('test_compile',3)
        result=difference(before)
        self.assertEqual(result['seconds']['test_compile'],5)
        self.assertEqual(result['counts']['test_compile'],2)
        self.assertIn('overlap',result['note'])
if __name__=='__main__': unittest.main()
