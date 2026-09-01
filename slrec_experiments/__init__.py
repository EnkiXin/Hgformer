"""Independent RecBole experiments for mixed-manifold recommendation.

The package deliberately does not import the recommender classes here.  Keeping
``__init__`` light allows the geometry utilities to be used without importing
RecBole (which is useful for tests and small numerical experiments).
"""

