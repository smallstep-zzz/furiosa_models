"""Registry for explicit public operator implementations after Furiosa's default compiler."""
from dataclasses import dataclass
from typing import Callable

_ORDER = {"python_decomposition": 0, "tcl": 1, "rust_visa": 2}
_ROUTES = {}


@dataclass(frozen=True)
class OperatorRoute:
    operator: str
    backend: str
    implementation: Callable
    supports: Callable


def register_operator(operator, backend, implementation, supports=lambda *a, **k: True):
    if backend not in _ORDER:
        raise ValueError(f"unknown extension backend: {backend}")
    route = OperatorRoute(operator, backend, implementation, supports)
    routes = _ROUTES.setdefault(operator, [])
    routes[:] = [item for item in routes if not (item.backend == backend and item.implementation is implementation)]
    routes.append(route)
    routes.sort(key=lambda item: _ORDER[item.backend])
    return implementation


def operator_routes(operator):
    return tuple(_ROUTES.get(operator, ()))


def load_builtin_routes():
    from . import graph_lowerings, rust_kernel_bridge, tcl_kernels  # noqa: F401


def resolve_operator(operator, *args, backend=None, **kwargs):
    load_builtin_routes()
    for route in operator_routes(operator):
        if (backend is None or route.backend == backend) and route.supports(*args, **kwargs):
            return route
    tried = ", ".join(route.backend for route in operator_routes(operator)) or "none"
    raise LookupError(f"no compatible route for {operator}; registered backends: {tried}")


def run_operator(operator, *args, backend=None, **kwargs):
    route = resolve_operator(operator, *args, backend=backend, **kwargs)
    return route.implementation(*args, **kwargs)
