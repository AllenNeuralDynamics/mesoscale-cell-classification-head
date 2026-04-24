"""Tests for :mod:`mesoscale_cell_classification_head.clustering`."""

from __future__ import annotations

import pytest
import torch

from mesoscale_cell_classification_head.clustering import OnlineKMeans

DEVICE = "cpu"
DIM = 16
K = 3


@pytest.fixture()
def kmeans() -> OnlineKMeans:
    """Return a fresh :class:`OnlineKMeans` instance on CPU."""
    return OnlineKMeans(n_clusters=K, dim=DIM, lr=0.1, device=DEVICE, stagnant_threshold=2)


def _rand_features(n: int = 20) -> torch.Tensor:
    return torch.randn(n, DIM)


class TestOnlineKMeansUpdate:
    def test_returns_labels_and_dists(self, kmeans: OnlineKMeans) -> None:
        x = _rand_features()
        labels, dists = kmeans.update(x)
        assert labels.shape == (len(x),)
        assert dists.shape == (len(x),)

    def test_labels_in_range(self, kmeans: OnlineKMeans) -> None:
        labels, _ = kmeans.update(_rand_features())
        assert labels.min() >= 0
        assert labels.max() < K

    def test_counts_increase(self, kmeans: OnlineKMeans) -> None:
        kmeans.update(_rand_features(30))
        assert kmeans.counts.sum() > 0

    def test_global_points_tracked(self, kmeans: OnlineKMeans) -> None:
        import numpy as np

        x = _rand_features(30)
        gp = np.random.randint(0, 100, size=(30, 3))
        kmeans.update(x, global_points=gp)
        # At least one cluster should have a best point
        assert any(p is not None for p in kmeans.best_points)

    def test_dead_centroid_reseeded(self, kmeans: OnlineKMeans) -> None:
        """Force stagnation by sending all points to one cluster region."""
        # Pack all features near centroid 0's current direction
        direction = kmeans.centroids[0].clone().unsqueeze(0).expand(20, -1)
        x = direction + 0.01 * torch.randn(20, DIM)
        for _ in range(kmeans.stagnant_threshold + 1):
            kmeans.update(x)
        # stagnant_counts for dead centroids should have been reset
        assert kmeans.stagnant_counts.max() == 0


class TestOnlineKMeansPredict:
    def test_shape(self, kmeans: OnlineKMeans) -> None:
        x = _rand_features(10)
        labels = kmeans.predict(x)
        assert labels.shape == (10,)

    def test_labels_in_range(self, kmeans: OnlineKMeans) -> None:
        labels = kmeans.predict(_rand_features(10))
        assert labels.min() >= 0
        assert labels.max() < K

    def test_centroids_unchanged(self, kmeans: OnlineKMeans) -> None:
        before = kmeans.centroids.clone()
        kmeans.predict(_rand_features(10))
        assert torch.allclose(before, kmeans.centroids)


class TestStateDict:
    def test_round_trip(self, kmeans: OnlineKMeans) -> None:
        kmeans.update(_rand_features(20))
        state = kmeans.state_dict()

        restored = OnlineKMeans(n_clusters=K, dim=DIM, device=DEVICE)
        restored.load_state_dict(state, device=DEVICE)

        assert torch.allclose(kmeans.centroids, restored.centroids)
        assert torch.allclose(kmeans.counts, restored.counts)

    def test_backwards_compat_missing_keys(self, kmeans: OnlineKMeans) -> None:
        """load_state_dict should tolerate old checkpoints missing new keys."""
        state = kmeans.state_dict()
        del state["stagnant_counts"]
        del state["stagnant_threshold"]

        restored = OnlineKMeans(n_clusters=K, dim=DIM, device=DEVICE)
        restored.load_state_dict(state, device=DEVICE)
        assert restored.stagnant_threshold == 50
