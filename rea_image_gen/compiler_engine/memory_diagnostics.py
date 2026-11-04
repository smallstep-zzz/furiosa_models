"""Bounded public-API checks. Logical tensor bytes are not allocator usage."""
import json,subprocess,time
from pathlib import Path
import torch

def _value(obj, name):
    value = getattr(obj, name)
    return value() if callable(value) else value

def smi_memory():
    result={}
    try:
        import furiosa_smi
        result['python_smi']={str(d):
            {str(core):{'total_bytes':_value(v,'total_bytes'),'in_use_bytes':_value(v,'in_use_bytes')} for core,v in d.memory_usage().items()}
            for d in furiosa_smi.list_devices()}
    except Exception as e:result['python_smi_error']=str(e)
    try:
        r=subprocess.run(['furiosa-smi','status','--format','json'],capture_output=True,text=True,timeout=5,check=True)
        result['cli']=json.loads(r.stdout)
    except Exception as e:result['cli_error']=str(e)
    return result

def inspect_memory(pipe,path,probe=False):
    started=time.perf_counter();rows={};seen=set()
    for name in ('unet','vae','text_encoder','text_encoder_2'):
        module=getattr(pipe,name,None)
        if module is None:continue
        by_device={}
        for kind,items in [('parameter',module.named_parameters()),('buffer',module.named_buffers())]:
            for _,t in items:
                if id(t) in seen:continue
                seen.add(id(t));key=str(t.device)
                group=by_device.setdefault(key,{'parameter_logical_bytes':0,'buffer_logical_bytes':0,'tensors':0})
                group[kind+'_logical_bytes']+=t.numel()*t.element_size();group['tensors']+=1
        rows[name]=by_device
    result={'components':rows,'telemetry':smi_memory(),'allocator_bytes':None,
        'residency_verdict':'SDK tensors bound to reported devices; exact physical HBM residency and executable-owned weight allocations remain unmeasured',
        'logical_bytes_note':'Unique Python tensor objects only; aliases/storage, padding, replicas and compiled constants can differ. This is NOT allocator memory usage.',
        'sdk_allocator_api':'No public memory_allocated/memory_reserved/memory_stats found in installed Torch SDK',
        'hypothesis':'SMI/driver/runtime allocation accounting path is not reporting these allocations; exact cause unconfirmed. CLI/libsmi2026.1.1 vs driver2026.4.0; Python SMI2026.4.0 also reports zero.'}
    if probe:
        # Deliberately no TCL kernel: a plain H2D allocation and D2H copy.
        cpu=torch.full((4*1024*1024,),0.125,dtype=torch.float32)
        tensor=cpu.to('furiosa:0')
        result['non_tcl_control']={'logical_bytes':tensor.numel()*tensor.element_size(),'device':str(tensor.device),
            'roundtrip_equal':bool(torch.equal(tensor.to('cpu'),cpu)),
            'telemetry_while_tensor_alive':smi_memory(), 'tcl_used':False}
        del tensor,cpu
    result['diagnostic_elapsed_s']=time.perf_counter()-started
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    path.with_suffix('.md').write_text('# NPU memory investigation\n\n'+result['residency_verdict']+'\n\n'+result['hypothesis']+'\n\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n')
    return result
