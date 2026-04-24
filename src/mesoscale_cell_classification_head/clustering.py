"""Online k-means clustering on GPU tensors.

The :class:`OnlineKMeans` class implements cosine-similarity-based k-means
with EMA centroid updates.  Dead centroids are automatically reseeded by
splitting the largest active cluster, which keeps all slots occupied
throughout training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


class OnlineKMeans:
    """Online k-means clustering with EMA updates and dead-centroid reseeding.

    Centroids live on a GPU device and are updated incrementally one
    mini-batch at a time.  Cosine similarity is used as the proximity
    measure, so all vectors are L2-normalised before assignment.

    Parameters
    ----------
    n_clusters : int
        Number of cluster centroids ``k``.
    dim : int
        Feature dimensionality.
    lr : float
        EMA learning rate for centroid updates.  Smaller values make
        centroids move more slowly.
    device : str
        PyTorch device string (e.g. ``"cuda"`` or ``"cpu"``).
    stagnant_threshold : int
        Number of consecutive batches a centroid may receive zero
        assignments before it is reseeded by splitting the largest cluster.
    """

    def __init__(
        self,
        n_clusters: int,
        dim: int,
        lr: float = 0.05,
        device: str = "cuda",
        stagnant_threshold: int = 50,
    ) -> None:
        self.k = n_clusters
        self.lr = lr
        self.device = device
        self.centroids: torch.Tensor = F.normalize(
            torch.randn(n_clusters, dim, device=device), dim=1
        )
        self.counts: torch.Tensor = torch.zeros(n_clusters, device=device)
        self.best_points: list[np.ndarray | None] = [None] * n_clusters
        self.best_dists: torch.Tensor = torch.full(
            (n_clusters,), float("inf"), device=device
        )
        self.stagnant_counts: torch.Tensor = torch.zeros(n_clusters, device=device)
        self.stagnant_threshold = stagnant_threshold

    @torch.no_grad()
    def update(
        self,
        x: torch.Tensor,
        global_points: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Assign ``x`` to clusters and update centroids via EMA.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, D)`` feature matrix on GPU.
        global_points : np.ndarray, optional
            ``(N, 3)`` world coordinates of the feature vectors, used to
            track the best representative point (medoid) per cluster.

        Returns
        -------
        labels : torch.Tensor
            ``(N,)`` cluster assignment indices.
        dists : torch.Tensor
            ``(N,)`` per-sample cosine distances to the assigned centroid.
        """
        x = F.normalize(x, dim=1)
        centroids_norm = F.normalize(self.centroids, dim=1)

        sim = x @ centroids_norm.T  # (N, k)
        labels = sim.argmax(dim=1)
        dists = 1 - sim[torch.arange(len(x)), labels]

        for i in range(self.k):
            mask = labels == i
            if not mask.any():
                self._handle_dead_centroid(i)
                continue

            self.stagnant_counts[i] = 0
            cluster_x = x[mask]

            updated = (1 - self.lr) * self.centroids[i] + self.lr * cluster_x.mean(dim=0)
            self.centroids[i] = F.normalize(updated, dim=0)
            self.counts[i] += mask.sum()

            if global_points is not None:
                self._update_best_point(i, mask, dists, global_points)

        return labels, dists

    def _handle_dead_centroid(self, i: int) -> None:
        """Reseed centroid ``i`` by splitting the largest active cluster.

        Parameters
        ----------
        i : int
            Index of the dead centroid to reseed.
        """
        self.stagnant_counts[i] += 1
        if self.stagnant_counts[i] < self.stagnant_threshold:
            return

        largest = int(self.counts.argmax().item())
        noise = F.normalize(torch.randn_like(self.centroids[largest]), dim=0)
        self.centroids[i] = F.normalize(self.centroids[largest] + 0.1 * noise, dim=0)
        self.counts[i] = self.counts[largest] // 2
        self.counts[largest] = self.counts[largest] // 2
        self.stagnant_counts[i] = 0

    def _update_best_point(
        self,
        i: int,
        mask: torch.Tensor,
        dists: torch.Tensor,
        global_points: np.ndarray,
    ) -> None:
        """Update the closest representative point for cluster ``i``.

        Parameters
        ----------
        i : int
            Cluster index.
        mask : torch.Tensor
            Boolean mask selecting the members of cluster ``i``.
        dists : torch.Tensor
            ``(N,)`` per-sample cosine distances to their assigned centroid.
        global_points : np.ndarray
            ``(N, 3)`` world coordinates of all feature vectors in the batch.
        """
        cluster_indices = mask.nonzero(as_tuple=False).squeeze(1).cpu().numpy()
        cluster_dists = dists[mask]
        min_dist, local_idx = cluster_dists.min(0)

        if min_dist.item() < self.best_dists[i].item():
            self.best_dists[i] = min_dist
            self.best_points[i] = global_points[cluster_indices[local_idx]]

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Assign cluster labels without updating centroids.

        Parameters
        ----------
        x : torch.Tensor
            ``(N, D)`` feature matrix on GPU.

        Returns
        -------
        torch.Tensor
            ``(N,)`` cluster assignment indices.
        """
        x = F.normalize(x, dim=1)
        centroids_norm = F.normalize(self.centroids, dim=1)
        sim = x @ centroids_norm.T
        return sim.argmax(dim=1)

    def state_dict(self) -> dict[str, Any]:
        """Serialise the model state to a plain dictionary.

        Returns
        -------
        dict[str, Any]
            All state tensors are detached and moved to CPU so the
            dictionary can be saved with :func:`torch.save`.
        """
        return {
            "k": self.k,
            "lr": self.lr,
            "centroids": self.centroids.detach().cpu(),
            "counts": self.counts.detach().cpu(),
            "best_points": self.best_points,
            "best_dists": self.best_dists.detach().cpu(),
            "stagnant_counts": self.stagnant_counts.detach().cpu(),
            "stagnant_threshold": self.stagnant_threshold,
        }

    def load_state_dict(self, state: dict[str, Any], device: str = "cuda") -> None:
        """Restore model state from a dictionary produced by :meth:`state_dict`.

        Parameters
        ----------
        state : dict[str, Any]
            Dictionary previously returned by :meth:`state_dict`.
        device : str
            Device to load tensors onto.
        """
        self.k = state["k"]
        self.lr = state["lr"]
        self.centroids = state["centroids"].to(device)
        self.counts = state["counts"].to(device)
        self.best_points = state["best_points"]
        self.best_dists = state["best_dists"].to(device)
        self.stagnant_counts = state.get(
            "stagnant_counts", torch.zeros(self.k)
        ).to(device)
        self.stagnant_threshold = state.get("stagnant_threshold", 50)
