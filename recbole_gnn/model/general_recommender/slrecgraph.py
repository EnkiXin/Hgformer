"""SL(n) graph collaborative filtering for the Hgformer runner.

This is the RecBole-GNN entry point for the implementation maintained in
``slrec_experiments.slrec``.  The model deliberately does not use PyG message
passing: it applies parameter-free LightGCN propagation with
``torch.sparse.mm`` to trace-free Lie-algebra coordinates and maps the result
to :math:`SL(n)` only for scoring.

Keeping this small adapter separate has two useful properties:

* ``run_recbole_gnn.py -m SLRecGraph`` discovers the model through the legacy
  RecBole-GNN naming convention (``SLRecGraph`` -> ``slrecgraph.py``).
* the controlled RecBole 1.2.x experiments and the Hgformer-protocol runs use
  exactly the same geometry and loss implementation.
"""

from slrec_experiments.slrec import SLRec


class SLRecGraph(SLRec):
    """Legacy-runner adapter for tangent-coordinate SL(n) graph CF.

    The raw entity width is ``num_factors * matrix_dim**2`` and the intrinsic
    trace-free dimension is ``num_factors * (matrix_dim**2 - 1)``.  Thus both
    ``SL(8)`` and ``SL(4)^4`` match the 64-scalar entity table of the Hgformer
    baselines (with 63 and 60 intrinsic dimensions, respectively); graph
    propagation itself has no parameters.  The optional learned score scale
    adds one global scalar, which can be disabled with
    ``learnable_score_scale: false`` for a literally equal raw-parameter entity
    budget.
    """

    def __init__(self, config, dataset):
        # The legacy trainer has analysis-specific return conventions.  Keep
        # the flags here rather than changing the shared SLRec implementation.
        self._tail_analysis = bool(config["tail_analysis"])
        self._popularity_analysis = bool(config["popularity_analysis"])
        super(SLRecGraph, self).__init__(config, dataset)

        baseline_dim = config["embedding_size"]
        if baseline_dim is not None and int(baseline_dim) != self.coordinate_dim:
            self.logger.warning(
                "SLRecGraph SL(%d)^%d uses %d raw coordinates per entity "
                "(%d intrinsic), but embedding_size=%s was supplied. Set "
                "embedding_size=%d when reporting an equal raw "
                "entity-embedding budget.",
                self.matrix_dim,
                self.num_factors,
                self.coordinate_dim,
                self.intrinsic_dim,
                baseline_dim,
                self.coordinate_dim,
            )

    def _analysis_result(self, scores):
        if self._tail_analysis:
            return self.head_item, self.tail_item, scores
        if self._popularity_analysis:
            return (
                self.rank1item,
                self.rank2item,
                self.rank3item,
                self.rank4item,
                self.rank5item,
                scores,
            )
        return scores

    def full_sort_predict(self, interaction):
        return self._analysis_result(
            super(SLRecGraph, self).full_sort_predict(interaction)
        )

    def full_sort_predict_with_exclusions(self, interaction, history_index):
        return self._analysis_result(
            super(SLRecGraph, self).full_sort_predict_with_exclusions(
                interaction, history_index
            )
        )


__all__ = ["SLRecGraph"]
