from importlib import import_module as _import_module

_impl = _import_module("llama3_neuroplastic.experiments.init_learned_basis_from_dense_mlp")
globals().update({name: getattr(_impl, name) for name in dir(_impl) if not name.startswith("__")})
__all__ = [name for name in globals() if not name.startswith("__")]
