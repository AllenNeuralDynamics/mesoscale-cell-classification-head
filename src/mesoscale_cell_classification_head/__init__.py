"""Mesoscale cell classification head package.

Public API
----------
- :class:`OnlineKMeans` — GPU online k-means with EMA updates
- :func:`greedy_cover_gpu` — greedy density-first box cover
- :func:`run_batch` — MAE encoder forward pass on image chunks
- :func:`extract_feature_vectors_torch` — per-cell feature extraction
- :func:`pad_to_shape` — 3-D array padding utility
"""

from mesoscale_cell_classification_head.clustering import OnlineKMeans
from mesoscale_cell_classification_head.feature_extraction import (
    extract_feature_vectors_torch,
    pad_to_shape,
    run_batch,
)
from mesoscale_cell_classification_head.spatial import greedy_cover_gpu

__version__ = "0.1.0"

__all__ = [
    "OnlineKMeans",
    "greedy_cover_gpu",
    "run_batch",
    "extract_feature_vectors_torch",
    "pad_to_shape",
]
