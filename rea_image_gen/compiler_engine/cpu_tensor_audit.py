"""Concrete CPU tensor audit for warm inference; fake/meta tensors excluded."""
from collections import Counter
import json
from pathlib import Path
import traceback
import torch
from torch.utils._pytree import tree_flatten

class ConcreteCPUAudit:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = {}
    def call(self, func, args=(), kwargs=None):
        kwargs = kwargs or {}
        tensors = [t for t in tree_flatten((args, kwargs))[0] if isinstance(t, torch.Tensor)]
        cpu = [t for t in tensors if type(t) is torch.Tensor and t.device.type == 'cpu']
        if cpu:
            stack = traceback.extract_stack(limit=24)[:-1]
            frames = [f'{f.filename}:{f.lineno}:{f.name}' for f in stack]
            key = (str(func), tuple(frames))
            if key not in self.rows:
                self.rows[key] = {'op': str(func), 'count': 0,
                    'inputs': [{'shape': list(t.shape), 'dtype': str(t.dtype)} for t in cpu],
                    'stack': frames, 'verdict': 'concrete_cpu_tensor_op_requires_callsite_review'}
            self.rows[key]['count'] += 1
        return func(*args, **kwargs)
    def save(self):
        rows = sorted(self.rows.values(), key=lambda r: -r['count'])
        self.path.write_text(json.dumps({'concrete_cpu_events': sum(r['count'] for r in rows),
            'callsites': rows}, indent=2))

    def __enter__(self):
        import functools
        import torch.nn.functional as F
        self.originals = []
        def wrap(owner, name):
            original = getattr(owner, name, None)
            if original is None: return
            @functools.wraps(original)
            def observed(*args, **kwargs):
                return self.call(original, args, kwargs)
            self.originals.append((owner, name, original))
            setattr(owner, name, observed)
        for name in ("linear", "conv2d", "group_norm", "layer_norm", "embedding", "scaled_dot_product_attention", "pad", "interpolate", "gelu", "silu", "softmax"):
            wrap(F, name)
        for name in ("add", "mul", "matmul", "bmm", "cat", "stack", "where", "arange", "randn", "zeros", "ones"):
            wrap(torch, name)
        return self
    def __exit__(self, *exc):
        for owner, name, original in reversed(self.originals):
            setattr(owner, name, original)
