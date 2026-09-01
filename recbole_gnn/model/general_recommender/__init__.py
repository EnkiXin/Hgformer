"""General graph recommenders with dependency-safe lazy imports.

Importing a concrete model should not require optional compiled operators used by
unrelated models. Attribute access keeps the original public import surface, for
example ``from ...general_recommender import LightGCN``.
"""

from importlib import import_module


_MODELS = {
    'LightGCN': 'lightgcn',
    'LHGCN': 'lhgcn',
    'SL8LHGCN': 'sl8lhgcn',
    'SL16LHGCN': 'sl16lhgcn',
    'AGCF': 'agcf',
    'AGCFSL8Coord': 'agcfsl8coord',
    'GGCF': 'ggcf',
    'HMLET': 'hmlet',
    'NCL': 'ncl',
    'NGCF': 'ngcf',
    'SGL': 'sgl',
    'LightGCL': 'lightgcl',
    'SimGCL': 'simgcl',
    'XSimGCL': 'xsimgcl',
    'DirectAU': 'directau',
    'SSL4REC': 'ssl4rec',
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
