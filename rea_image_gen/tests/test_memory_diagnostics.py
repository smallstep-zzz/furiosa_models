import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import torch
from compiler_engine.memory_diagnostics import inspect_memory
class MemoryTests(unittest.TestCase):
 def test_logical_bytes_are_not_allocator_bytes(self):
  pipe=SimpleNamespace(unet=torch.nn.Linear(2,3))
  with tempfile.TemporaryDirectory() as d,patch('compiler_engine.memory_diagnostics.smi_memory',return_value={'test':True}):
   r=inspect_memory(pipe,Path(d)/'memory.json')
   self.assertEqual(r['components']['unet']['cpu']['parameter_logical_bytes'],36)
   self.assertIsNone(r['allocator_bytes'])
   self.assertNotIn('non_tcl_control',r)
   self.assertTrue((Path(d)/'memory.md').exists())
if __name__=='__main__':unittest.main()
