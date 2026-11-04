"""Per-process measurements; development history is explicit, never inferred."""
from collections import defaultdict
from contextlib import contextmanager
import time

_totals=defaultdict(float)
_counts=defaultdict(int)

def record(kind, seconds=0):
    _totals[kind]+=seconds
    _counts[kind]+=1

def snapshot():
    return dict(_totals),dict(_counts)

def difference(before):
    times,counts=before
    return {'seconds':{k:round(v-times.get(k,0),6) for k,v in _totals.items()},
            'counts':{k:v-counts.get(k,0) for k,v in _counts.items()},
            'note':'Nested phase totals overlap; do not add them to wall time. Inference wall time includes host scheduler, transfers and any lazy compilation; it is not pure NPU kernel time.'}

@contextmanager
def phase(name):
    started=time.perf_counter()
    try: yield
    finally: record(name,time.perf_counter()-started)
