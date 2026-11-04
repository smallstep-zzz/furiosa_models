import torch
from resident_model import adapt as baseline_adapt
from pointwise_model import DirectPointwise

def adapt(model):
 originals=[(name,m) for name,m in model.named_modules() if isinstance(m,torch.nn.Conv2d) and m.kernel_size==(1,1) and m.out_channels!=1]
 first=model.get_submodule('model/0/conv/Conv')
 model=baseline_adapt(model)
 for name,m in originals:model.set_submodule(name,DirectPointwise(m,name))
 from first_layout import FirstConv
 model.set_submodule('model/0/conv/Conv',FirstConv(first,'model/0/conv/Conv'))
 return model
