"""Conservatively fold sole-use Conv -> Sigmoid/Mul activation patterns."""
import torch


def matches(model):
    from onnx2torch.node_converters.binary_math_operations import OnnxBinaryMathOperation
    calls = {}
    for node in model.graph.nodes:
        if node.op == 'call_module':
            calls[node.target] = calls.get(node.target, 0) + 1
    found = []
    for conv in model.graph.nodes:
        if conv.op != 'call_module' or calls[conv.target] != 1:
            continue
        module = model.get_submodule(conv.target)
        if not isinstance(module, torch.nn.Conv2d) or module.out_channels == 1:
            continue
        pointwise = (module.kernel_size == (1, 1) and module.stride == (1, 1)
                     and module.padding == (0, 0) and module.groups == 1)
        spatial = module.kernel_size == (3, 3) and module.in_channels != 3
        if not (pointwise or spatial) or module.dilation != (1, 1):
            continue
        if len(conv.users) != 2:
            continue
        for sigmoid in conv.users:
            if (sigmoid.op != 'call_module' or sigmoid.args != (conv,)
                    or sigmoid.kwargs or len(sigmoid.users) != 1
                    or type(model.get_submodule(sigmoid.target)) is not torch.nn.Sigmoid):
                continue
            mul = next(iter(sigmoid.users))
            if (mul.op != 'call_module' or mul not in conv.users or mul.kwargs
                    or mul.args not in ((conv, sigmoid), (sigmoid, conv))):
                continue
            operation = model.get_submodule(mul.target)
            if (type(operation) is not OnnxBinaryMathOperation
                    or operation.math_op_function is not torch.mul
                    or (operation.broadcast == 1 and operation.axis is not None)):
                continue
            found.append((conv, sigmoid, mul, module, pointwise))
    return found


def adapt(model):
    from direct_route import adapt as direct_adapt
    from activation_pointwise import DirectPointwise
    from activation_fused import FusedConv
    selected = matches(model)
    model = direct_adapt(model)
    for conv, sigmoid, mul, original, pointwise in selected:
        wrapper = DirectPointwise if pointwise else FusedConv
        model.set_submodule(conv.target, wrapper(original, conv.target))
        mul.replace_all_uses_with(conv)
        model.graph.erase_node(mul)
        model.graph.erase_node(sigmoid)
    model.graph.lint()
    model.recompile()
    model.activation_fusion_count = len(selected)
    return model
