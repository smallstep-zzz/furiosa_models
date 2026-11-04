import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location('npu_watch',Path(__file__).parents[1]/'tools/npu_watch.py')
watch=importlib.util.module_from_spec(spec);spec.loader.exec_module(watch)
class WatchTests(unittest.TestCase):
    def test_ratio_units_and_report(self):
        raw=[{'device':'npu2','liveness':'alive','memory':{'DRAM':{'used_size':2**30,'total_size':48*2**30}},'pe_utilizations':[{'pe_core':0,'pe_utilization':0.25}]}]
        rows=watch.normalize(raw)
        self.assertEqual(rows[0]['cores_pct']['0'],0.25)
        with tempfile.TemporaryDirectory() as d:
            result=watch.report([{'devices':rows},{'error':'timeout'}],Path(d))
            self.assertEqual(result['errors'],1)
            self.assertEqual(result['devices']['npu2']['peak_core_pct'],0.25)
            self.assertTrue((Path(d)/'REPORT.md').exists())
if __name__=='__main__': unittest.main()
