"""Tests for :mod:`mesoscale_cell_classification_head.pipeline`."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mesoscale_cell_classification_head.pipeline import (
    _cap_features,
    _load_box_batch,
    infer,
    train,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_boxes(
    n: int = 4,
    box_size: int = 64,
) -> tuple[list, list]:
    """Return ``n`` non-overlapping boxes and trivial cell-index lists."""
    boxes = []
    box_cells = []
    for i in range(n):
        start = np.array([i * box_size, 0, 0])
        end = np.array([(i + 1) * box_size, box_size, box_size])
        boxes.append((start, end))
        box_cells.append(np.array([i], dtype=np.int64))
    return boxes, box_cells


def _make_zarr_mock(shape: tuple[int, ...]) -> MagicMock:
    arr = np.zeros(shape, dtype=np.float32)
    mock = MagicMock()
    mock.__getitem__ = MagicMock(return_value=MagicMock(compute=MagicMock(return_value=arr)))
    return mock


# ---------------------------------------------------------------------------
# _cap_features
# ---------------------------------------------------------------------------

class TestCapFeatures:
    def test_no_cap_needed(self) -> None:
        feats = torch.rand(10, 4)
        gp = np.zeros((10, 3))
        out_f, out_g = _cap_features(feats, gp, max_cells=20, device=torch.device("cpu"))
        assert out_f.shape == feats.shape
        assert len(out_g) == 10

    def test_cap_applied(self) -> None:
        feats = torch.rand(50, 4)
        gp = np.zeros((50, 3))
        out_f, out_g = _cap_features(feats, gp, max_cells=10, device=torch.device("cpu"))
        assert out_f.shape[0] == 10
        assert len(out_g) == 10


# ---------------------------------------------------------------------------
# _load_box_batch
# ---------------------------------------------------------------------------

class TestLoadBoxBatch:
    def test_skips_empty_boxes(self) -> None:
        boxes = [(np.array([0, 0, 0]), np.array([8, 8, 8]))]
        box_cells = [np.array([], dtype=np.int64)]
        cell_zyx = np.zeros((0, 3), dtype=np.int64)
        zarr = _make_zarr_mock((64, 64, 64))

        chunks, points, starts = _load_box_batch([0], boxes, box_cells, cell_zyx, zarr)
        assert chunks == []

    def test_loads_valid_box(self) -> None:
        cell_zyx = np.array([[4, 4, 4]], dtype=np.int64)
        boxes = [(np.array([0, 0, 0]), np.array([8, 8, 8]))]
        box_cells = [np.array([0], dtype=np.int64)]
        zarr = _make_zarr_mock((64, 64, 64))

        chunks, points, starts = _load_box_batch([0], boxes, box_cells, cell_zyx, zarr)
        assert len(chunks) == 1
        assert chunks[0].shape == (8, 8, 8)


# ---------------------------------------------------------------------------
# train / infer — smoke tests with mocked model
# ---------------------------------------------------------------------------

def _make_reconstruction_model(n_cells: int, n_pca: int = 64) -> MagicMock:
    """Return a mock MAE model producing feature maps of shape (B, C, 32, 32, 32)."""
    C = n_pca  # use same dim so PCA round-trip works
    n_skip = 2
    n_tokens = 32 * 32 * 32 + n_skip

    def encoder_side_effect(batch, mask_ratio, recover_layers):
        B = batch.shape[0]
        return (torch.zeros(B, n_tokens, C), None, None, None, None, None)

    mock_encoder = MagicMock(side_effect=encoder_side_effect)
    mock = MagicMock()
    mock.model.encoder = mock_encoder
    return mock


@pytest.mark.parametrize("n_boxes", [1, 3])
def test_train_returns_ipca(n_boxes: int, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    N_PCA = 8
    boxes, box_cells = _make_boxes(n=n_boxes, box_size=128)
    cell_zyx = np.array([[64, 64, 64]] * n_boxes, dtype=np.int64)
    zarr = _make_zarr_mock((512, 128, 128))
    model = _make_reconstruction_model(n_cells=n_boxes, n_pca=N_PCA)
    device = torch.device("cpu")

    cfg = {
        "box_size": None,
        "vox": 32,
        "scale": 1,
        "max_boxes": n_boxes,
        "batch_size": n_boxes,
        "jump": 8,
        "n_clusters": 2,
        "n_pca_components": N_PCA,
        "max_cells_per_box": 128,
        "ipca_warmup_boxes": 0,   # skip warmup so kmeans is always initialised
        "store_kmeans_every_n": 9999,
        "lr_kmeans": 0.1,
    }

    ipca, kmeans = train(boxes, box_cells, cell_zyx, zarr, model, device, cfg)
    assert ipca is not None


def test_infer_returns_arrays(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)

    from sklearn.decomposition import IncrementalPCA

    from mesoscale_cell_classification_head.clustering import OnlineKMeans

    N_PCA = 8
    boxes, box_cells = _make_boxes(n=2, box_size=128)
    cell_zyx = np.array([[64, 64, 64], [64, 64, 64]], dtype=np.int64)
    zarr = _make_zarr_mock((512, 128, 128))
    model = _make_reconstruction_model(n_cells=2, n_pca=N_PCA)
    device = torch.device("cpu")

    # Fit a real IPCA on random data so transform works
    ipca = IncrementalPCA(n_components=N_PCA, whiten=True)
    ipca.partial_fit(np.random.rand(N_PCA + 5, N_PCA))

    kmeans = OnlineKMeans(n_clusters=2, dim=N_PCA, device="cpu")

    cfg = {
        "max_boxes": 2,
        "batch_size": 2,
        "jump": 8,
        "n_pca_components": N_PCA,
    }

    points, labels = infer(boxes, box_cells, cell_zyx, zarr, model, device, ipca, kmeans, cfg)
    assert points.ndim == 2
    assert labels.ndim == 1
    assert points.shape[0] == labels.shape[0]
