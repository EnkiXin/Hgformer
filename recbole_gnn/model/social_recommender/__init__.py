"""Social graph recommenders with dependency-safe lazy imports."""

from importlib import import_module


_MODELS = {
    'DiffNet': 'diffnet',
    'MHCN': 'mhcn',
    'SEPT': 'sept',
}

__all__ = list(_MODELS)


def __getattr__(name):
    try:
        module_name = _MODELS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    model = getattr(import_module(f'{__name__}.{module_name}'), name)
    globals()[name] = model
    return model
