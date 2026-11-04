"""Specialize ONNX constant shape/index metadata before torch.export."""
import operator
import hashlib
import json
from pathlib import Path
import torch
from onnx2torch.node_converters.reshape import OnnxReshape
from onnx2torch.node_converters.resize import OnnxResize
from onnx2torch.node_converters.slice import OnnxSlice


def _apply(model, replacements):
    for old, (target, args, kwargs) in reversed(list(replacements.items())):
        with model.graph.inserting_before(old):
            new = model.graph.call_function(target, args=args, kwargs=kwargs)
        old.replace_all_uses_with(new)
        model.graph.erase_node(old)
    model.graph.eliminate_dead_code()
    model.graph.lint()
    model.recompile()
    return model


def specialize_saved(model, input_shape, metadata_path):
    """Reconstruct a validated static graph without any CPU model forward."""
    metadata = json.loads(Path(metadata_path).read_text())
    digest = hashlib.sha256(str(model.graph).encode()).hexdigest()
    if metadata['input_shape'] != list(input_shape) or metadata['graph_sha256'] != digest:
        raise ValueError('Static ONNX metadata shape/graph mismatch')
    nodes = {n.name: n for n in model.graph.nodes}
    replacements = {}
    for row in metadata['replacements']:
        node = nodes[row['node']]
        if row['kind'] == 'reshape':
            replacements[node] = (torch.reshape, (node.args[0], tuple(row['shape'])), {})
        elif row['kind'] == 'resize':
            replacements[node] = (torch.nn.functional.interpolate, (node.args[0],),
                                  {'size': tuple(row['shape']), 'mode': 'nearest'})
        elif row['kind'] == 'slice':
            replacements[node] = (operator.getitem, (node.args[0], tuple(slice(*s) for s in row['indices'])), {})
        else:
            raise ValueError('Unknown static ONNX operation')
    return _apply(model, replacements)


def specialize(model, example, metadata_path=None):
    replacements = {}

    class Capture(torch.fx.Interpreter):
        def run_node(self, node):
            result = super().run_node(node)
            if node.op != 'call_module':
                return result
            module = self.fetch_attr(node.target)
            if not isinstance(module, (OnnxReshape, OnnxResize, OnnxSlice)):
                return result
            # Only initializer metadata may become constants; model data stays in the graph.
            assert all(n.op == 'get_attr' for n in node.all_input_nodes[1:])
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            if isinstance(module, OnnxReshape):
                replacements[node] = (torch.reshape, (node.args[0], tuple(result.shape)), {})
            elif isinstance(module, OnnxResize):
                assert module.onnx_mode == 'nearest'
                replacements[node] = (torch.nn.functional.interpolate, (node.args[0],),
                                      {'size': tuple(result.shape[2:]), 'mode': 'nearest'})
            else:
                starts, ends = args[1].tolist(), args[2].tolist()
                axes = args[3].tolist() if len(args) > 3 and args[3] is not None else list(range(len(starts)))
                steps = args[4].tolist() if len(args) > 4 and args[4] is not None else [1]*len(starts)
                indices = [slice(None)]*args[0].ndim
                for start, end, axis, step in zip(starts, ends, axes, steps):
                    assert step > 0, 'Negative slicing requires a separate verified adapter'
                    indices[axis] = slice(start, end, step)
                replacements[node] = (operator.getitem, (node.args[0], tuple(indices)), {})
            return result

    with torch.no_grad():
        reference = Capture(model).run(example)
    if metadata_path:
        rows = []
        for node, (target, args, kwargs) in replacements.items():
            if target is torch.reshape:
                rows.append({'node': node.name, 'kind': 'reshape', 'shape': list(args[1])})
            elif target is torch.nn.functional.interpolate:
                rows.append({'node': node.name, 'kind': 'resize', 'shape': list(kwargs['size'])})
            else:
                rows.append({'node': node.name, 'kind': 'slice',
                             'indices': [[s.start, s.stop, s.step] for s in args[1]]})
        Path(metadata_path).write_text(json.dumps({'input_shape': list(example.shape),
                'graph_sha256': hashlib.sha256(str(model.graph).encode()).hexdigest(),
                'replacements': rows}, indent=2))
    _apply(model, replacements)
    with torch.no_grad():
        torch.testing.assert_close(model(example), reference, rtol=0, atol=0)
    return model
