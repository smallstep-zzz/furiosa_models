#!/usr/bin/env python3
"""Watch public RNGD SMI JSON; retain samples and a compact utilization report."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import signal
import subprocess
import time


def normalize(devices):
    rows = []
    for d in devices:
        dram = d.get('memory', {}).get('DRAM', {})
        rows.append({'device': d['device'], 'liveness': d.get('liveness'),
            'memory_used_bytes': dram.get('used_size'), 'memory_total_bytes': dram.get('total_size'),
            'cores_pct': {str(p['pe_core']): float(p['pe_utilization'])
                          for p in d.get('pe_utilizations', [])}})
    return rows


def report(samples, output):
    devices = defaultdict(list)
    errors = 0
    for sample in samples:
        errors += int('error' in sample)
        for row in sample.get('devices', []): devices[row['device']].append(row)
    result = {'samples': len(samples), 'errors': errors, 'devices': {},
              'note': 'SMI sampling is device telemetry, not operator execution proof. Short bursts may be missed; memory may be reported as zero by SDK.'}
    lines = ['# RNGD utilization report', '', result['note'], '',
             f'Samples: {len(samples)}; errors: {errors}', '',
             '|Device|Samples|Mean core utilization|Peak core utilization|Peak memory GiB|',
             '|---|---:|---:|---:|---:|']
    for name, rows in devices.items():
        values = [v for r in rows for v in r['cores_pct'].values()]
        item = {'samples': len(rows), 'mean_core_pct': sum(values)/len(values) if values else None,
                'peak_core_pct': max(values) if values else None,
                'peak_memory_bytes': max((r['memory_used_bytes'] or 0 for r in rows), default=0)}
        result['devices'][name] = item
        lines.append(f"|{name}|{len(rows)}|{item['mean_core_pct'] or 0:.3f}%|{item['peak_core_pct'] or 0:.3f}%|{item['peak_memory_bytes']/2**30:.3f}|")
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    (output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval',type=float,default=0.5)
    parser.add_argument('--duration',type=float,default=0,help='Seconds; 0 means until Ctrl-C')
    parser.add_argument('--output',type=Path,default=Path('reports/npu_watch')/datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--no-clear',action='store_true')
    parser.add_argument('--quiet',action='store_true')
    parser.add_argument('--ps-interval',type=float,default=10)
    args=parser.parse_args()
    if args.interval<=0 or args.duration<0 or args.ps_interval<=0: parser.error('invalid interval/duration')
    if not shutil.which('furiosa-smi'): parser.error('furiosa-smi not found; run this tool on the RNGD server')
    args.output.mkdir(parents=True,exist_ok=True)
    stop=False
    def halt(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,halt);signal.signal(signal.SIGTERM,halt)
    started=time.monotonic(); next_ps=started;samples=[]
    try:
        with (args.output/'samples.jsonl').open('a',buffering=1) as log:
            while not stop:
                tick=time.monotonic()
                if args.duration and tick-started>=args.duration: break
                sample={'timestamp':datetime.now(timezone.utc).isoformat(),'elapsed_s':tick-started}
                try:
                    proc=subprocess.run(['furiosa-smi','status','--format','json'],capture_output=True,text=True,timeout=10,check=True)
                    sample['raw_status']=json.loads(proc.stdout)
                    sample['devices']=normalize(sample['raw_status'])
                    if tick>=next_ps:
                        ps=subprocess.run(['furiosa-smi','ps'],capture_output=True,text=True,timeout=10)
                        sample['process_mapping']=ps.stdout
                        sample['process_mapping_returncode']=ps.returncode
                        next_ps=tick+args.ps_interval
                except (subprocess.SubprocessError, ValueError, KeyError, TypeError) as exc:
                    sample['error']=str(exc)
                log.write(json.dumps(sample)+'\n');samples.append(sample)
                if not args.quiet:
                    if not args.no_clear: print('\033[2J\033[H',end='')
                    print(sample['timestamp'],f"elapsed={sample['elapsed_s']:.1f}s",flush=True)
                    for row in sample.get('devices',[]):
                        cores=' '.join(f'{k}:{v:.2f}%' for k,v in row['cores_pct'].items())
                        print(f"{row['device']} {row['liveness']} memory={(row['memory_used_bytes'] or 0)/2**30:.2f} GiB | {cores}",flush=True)
                    if 'error' in sample: print(sample['error'],flush=True)
                # Refresh compact reports every ten seconds; raw samples flush each tick.
                if len(samples)==1 or len(samples)%max(1,int(10/args.interval))==0: report(samples,args.output)
                wait=max(0,args.interval-(time.monotonic()-tick))
                if args.duration: wait=min(wait,max(0,args.duration-(time.monotonic()-started)))
                time.sleep(wait)
    finally:
        result=report(samples,args.output)
        print(json.dumps({'report':str(args.output/'REPORT.md'),'samples':result['samples'],'errors':result['errors']}),flush=True)

if __name__=='__main__': main()
