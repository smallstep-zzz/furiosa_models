import tempfile
import unittest
from pathlib import Path
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from compiler_engine.cpu_tensor_audit import ConcreteCPUAudit

class ConcreteAuditTests(unittest.TestCase):
    def test_real_cpu_detected_fake_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            audit=ConcreteCPUAudit(Path(d)/'audit.json')
            x=torch.ones(4)
            with audit:
                y=torch.add(x,2)
            self.assertTrue(any('add' in r['op'] for r in audit.rows.values()))
            audit2=ConcreteCPUAudit(Path(d)/'fake.json')
            with FakeTensorMode():
                fake=torch.ones(4)
                with audit2:
                    z=torch.add(fake,2)
            self.assertEqual(len(audit2.rows),0)
            audit.save()
            self.assertTrue(audit.path.exists())

if __name__=='__main__': unittest.main()
