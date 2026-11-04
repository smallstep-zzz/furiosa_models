"""CPU calculator/export verification; explicit --target furiosa for coordinator."""
import argparse
import json
import time
import torch
import torch.nn.functional as F
from tcl_conv_candidate import module, fused_conv, prepare_weights

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--target',choices=['cpu','furiosa'],default='cpu')
    p.add_argument('--shape',type=int,nargs=4,default=[1,16,320,320])
    p.add_argument('--out-channels',type=int,default=32)
    p.add_argument('--real-weights',action='store_true')
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(97)
    shape=tuple(args.shape)
    value=torch.randn(shape)
    weight=torch.randn(args.out_channels,shape[1],3,3)*0.1
    bias=torch.randn(args.out_channels)*0.1
    if args.real_weights:
        import onnx
        from onnx2torch import convert
        model=convert(onnx.load('/root/yolo11-rngd/models/yolo11n_640.onnx'))
        first=next(m for m in model.modules() if isinstance(m,torch.nn.Conv2d))
        if first.out_channels!=args.out_channels or first.in_channels>shape[1]:
            raise ValueError('Real first conv cannot fit selected geometry')
        weight.zero_()
        weight[:,:first.in_channels]=first.weight.detach()
        value[:,first.in_channels:]=0
        bias=first.bias.detach().clone() if first.bias is not None else torch.zeros(first.out_channels)
    reference=F.conv2d(value,weight,bias,padding=1,stride=2)
    weight_hi,weight_lo=prepare_weights(weight)
    inputs=(value,weight_hi,weight_lo,bias)
    started=time.perf_counter()
    m=module(shape,tuple(weight.shape))
    ep=torch.export.export(m,inputs,strict=False)
    print(json.dumps({'export_seconds':time.perf_counter()-started,'graph':str(ep.graph)}),flush=True)
    if args.target=='cpu':
        result=m(*inputs)[0]
        times=[]
    else:
        resident=tuple(x.to('furiosa:0') for x in inputs)
        result=fused_conv(*resident)
        result.cpu()  # finish warmup before timing
        times=[]
        for _ in range(5):
            started=time.perf_counter()
            result=fused_conv(*resident)
            result_cpu=result.cpu()  # synchronization includes output transfer
            times.append(time.perf_counter()-started)
        result=result_cpu
    torch.testing.assert_close(result,reference,rtol=1e-3,atol=1e-3)
    print(json.dumps({'target':args.target,'shape':list(result.shape),'dtype':str(result.dtype),
          'max_abs_error':(result-reference).abs().max().item(),'timings_with_output_copy_s':times,
          'npu_trace':'coordinator must collect official Task events' if args.target=='furiosa' else 'untested'}),flush=True)
if __name__=='__main__':
    main()
