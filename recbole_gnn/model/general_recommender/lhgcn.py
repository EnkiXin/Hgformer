"""Standalone Light Hyperbolic GCN for the legacy Hgformer runner.

The released Hgformer code does not contain a class named ``LHGCN``.  Its
parameter-free local convolution lives in :class:`HGCN` under the ``"lGCN"``
branch and is also available through ``HGCF`` by selecting ``conv: lGCN``.
This small adapter gives that archived standalone model an explicit RecBole
name without constructing Hgformer's cross-attention encoder.

Inheriting from :class:`HGCF` is intentional: it preserves the historical
combined user/item embedding table, hyperboloid initialisation, last-layer
``lGCN`` propagation, squared-distance margin-ranking loss, and full-ranking
negative-distance decoder exactly.  It should not be confused with
``RecFormer(no_transformer=True)``: that path uses separate embedding tables
and an unsquared-distance loss, and currently computes the Transformer before
discarding its output.
"""

from __future__ import annotations

from typing import Any

from recbole_gnn.model.general_recommender.hgcf import HGCF


class LHGCN(HGCF):
    """Expose the archived ``HGCF + conv=lGCN`` model as ``LHGCN``.

    ``conv`` is canonicalised to the spelling expected by ``HGCN``.  Supplying
    another convolution is rejected so an experiment labelled LHGCN cannot
    silently run the tangent-space ``resSumGCN`` or parameterised HGCN branch.
    All remaining model keys retain their HGCF meanings, notably
    ``gcn_layers``, ``curve``, ``margin``, ``scale``, and ``learner``.
    """

    input_type = HGCF.input_type

    def __init__(self, config: Any, dataset: Any) -> None:
        configured_conv = config["conv"]
        if configured_conv is None:
            config["conv"] = "lGCN"
        elif str(configured_conv).strip().lower() == "lgcn":
            # HGCN's archived dispatch is case-sensitive.
            config["conv"] = "lGCN"
        else:
            raise ValueError(
                "LHGCN requires conv='lGCN'; "
                f"received conv={configured_conv!r}"
            )
        super().__init__(config, dataset)


__all__ = ["LHGCN"]
