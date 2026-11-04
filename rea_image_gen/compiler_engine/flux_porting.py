"""Incremental FLUX feature routes; not a full-model support declaration."""
from .renegade_diffusion_compiler import _NPUBlock

def segment_adaptive_norm(module, component='transformer.norm'):
    # Keep supported primitives separate when the fused AdaLN graph fails axis planning.
    for name in ('silu', 'linear', 'norm'):
        child=getattr(module,name,None)
        if child is not None and not isinstance(child,_NPUBlock):
            setattr(module,name,_NPUBlock(child,'flux',component+'.'+name))
    return module
