"""Relocation acceptance under an inherited, kernel-enforced read allowlist."""
import ctypes,os,sys,json,time
from pathlib import Path
BASE=Path(__file__).resolve().parent
package=BASE/'yolo11n-portable.fxbex'
libc=ctypes.CDLL(None,use_errno=True);libc.syscall.restype=ctypes.c_long
class Ruleset(ctypes.Structure):_fields_=[('handled_access_fs',ctypes.c_uint64)]
class Beneath(ctypes.Structure):_fields_=[('allowed_access',ctypes.c_uint64),('parent_fd',ctypes.c_int)]
def check(result):
 if result<0:raise OSError(ctypes.get_errno(),os.strerror(ctypes.get_errno()))
 return result
abi=check(libc.syscall(444,0,0,1));mask=1|4|8;rule=Ruleset(mask);fd=check(libc.syscall(444,ctypes.byref(rule),ctypes.sizeof(rule),0))
roots=['/usr','/lib','/lib64','/bin','/sbin','/etc','/dev','/proc','/sys','/tmp','/run','/opt','/root/FXBEX','/root/rngd-model-forge/reports/rngd_porting/npu.lock']
cache=Path('/root/.cache')
if cache.exists():roots += [str(p) for p in cache.iterdir() if p.is_dir() and ('furiosa' in p.name.lower() or p.name=='torch')]
for name in roots:
 p=Path(name)
 if not p.exists():continue
 pf=os.open(p,os.O_PATH|os.O_CLOEXEC);entry=Beneath(mask if p.is_dir() else 1|4,pf)
 check(libc.syscall(445,fd,1,ctypes.byref(entry),0));os.close(pf)
check(libc.prctl(38,1,0,0,0));check(libc.syscall(446,fd,0));os.close(fd)
blocked=[]
for name in ['/root/yolo11-rngd/models/yolo11n_640.onnx','/root/yolo11-rngd-opt/resident_model.py','/root/yolo11-rngd-opt2/fused_model.py', '/root/yolo11-rngd-opt3/activation_route.py','/root/rngd-model-forge/compiler_engine/__init__.py']:
 try:open(name,'rb').read(1)
 except PermissionError:blocked.append(name)
 else:raise AssertionError('Original source still readable: '+name)
try:os.listdir('/root/.cache/huggingface')
except (PermissionError,FileNotFoundError):blocked.append('original HF cache')
else:raise AssertionError('HF cache remains accessible')
os.environ['FXBEX_CONFIG']=str(BASE/'fxbex_config.json')
os.environ['FXBEX_NPU_LOCK']='/root/rngd-model-forge/reports/rngd_porting/npu.lock'
os.environ['XDG_CACHE_HOME']=str(BASE/'cache')
os.environ['HF_HOME']=str(BASE/'empty_hf_cache')
os.environ['HF_HUB_OFFLINE']='1'
from fxbex import YOLO
results=[]
with YOLO(package) as model:
 for i in range(4):
  result=model.predict(BASE/'bus.jpg');assert result['compiler_calls']==0
  result['host_request_timing']=model.last_timing
  results.append(result);print(json.dumps({'i':i,'inference_s':result['inference_s'],'timing':model.last_timing,'result':result['output']}),flush=True)
report={'landlock_abi':abi,'denied_original_paths':blocked,'original_project_inaccessible':True,'results':results,'api':'fxbex.YOLO(package).predict(image)','model_unloaded':True}
(BASE/'isolated_result.json').write_text(json.dumps(report,indent=2));print('ISOLATED_PASSED',flush=True)
