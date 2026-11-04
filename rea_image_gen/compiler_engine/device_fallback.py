"""FX graph preparation for the CPU path used after an NPU compile rejection."""

import torch


def replace_graph_device(graph_module, device):
    """Replace explicit Furiosa device constants before executing a graph on CPU."""
    device = torch.device(device)

    def replace(value):
        if isinstance(value, torch.device) and value.type in ("rngd", "furiosa"):
            return device
        if isinstance(value, str) and value.startswith(("rngd", "furiosa")):
            return str(device)
        return value

    def replace_nested(value):
        value = replace(value)
        if isinstance(value, tuple):
            return tuple(replace_nested(item) for item in value)
        if isinstance(value, list):
            return [replace_nested(item) for item in value]
        if isinstance(value, dict):
            return {key: replace_nested(item) for key, item in value.items()}
        return value

    for node in graph_module.graph.nodes:
        node.args = replace_nested(node.args)
        node.kwargs = replace_nested(node.kwargs)
    graph_module.to(device)
    graph_module.graph.lint()
    graph_module.recompile()
    return graph_module
