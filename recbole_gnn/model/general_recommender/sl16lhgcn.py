"""Dimension-16 variant of the shared special-linear LHGCN implementation."""

from recbole_gnn.model.general_recommender.sl8lhgcn import SL8LHGCN


class SL16LHGCN(SL8LHGCN):
    """Final-layer LHGCN-style collaborative filtering in ``SL(16)``.

    All propagation, retraction, distance, loss, prediction and full-sort code
    is inherited from :class:`SL8LHGCN`.  Only the controlled manifold
    dimension changes: one 16x16 factor stores 256 raw coordinates and has
    255 trace-free degrees of freedom per entity.
    """

    MODEL_NAME = "SL16LHGCN"
    REQUIRED_MATRIX_DIM = 16
    REQUIRED_NUM_FACTORS = 1


__all__ = ["SL16LHGCN"]
