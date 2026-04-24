"""Tests for :mod:`mesoscale_cell_classification_head.feature_extraction`."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from mesoscale_cell_classification_head.feature_extraction import (
    extract_feature_vectors_torch,
    pad_to_shape,
    run_batch,
)


class TestPadToShape:
    def test_already_correct_shape(self) -> None:
        arr = np.ones((4, 4, 4))
        out = pad_to_shape(arr, target_shape=(4, 4, 4))
        assert out.shape == (4, 4, 4)
        np.testing.assert_array_equal(out, arr)

    def test_pads_correctly(self) -> None:
        arr = np.ones((2, 3, 4))
        out = pad_to_shape(arr, target_shape=(4, 4, 4))
        assert out.shape == (4, 4, 4)
        # Original data preserved at origin
        np.testing.assert_array_equal(out[:2, :3, :4], arr)
        # Padding is zero
        assert out[2, 0, 0] == 0

    def test_raises_on_wrong_ndim(self) -> None:
        with pytest.raises(AssertionError):
            pad_to_shape(np.ones((4, 4)), target_shape=(4, 4, 4))

    def test_raises_when_dim_exceeds_target(self) -> None:
        with pytest.raises(ValueError, match="exceeds target"):
            pad_to_shape(np.ones((8, 4, 4)), target_shape=(4, 4, 4))

    def test_constant_value(self) -> None:
        arr = np.zeros((2, 2, 2))
        out = pad_to_shape(arr, target_shape=(4, 4, 4), constant_values=99)
        assert out[3, 3, 3] == 99


def _make_mock_model(feature_shape: tuple[int, int, int, int, int]) -> MagicMock:
    """Return a mock MAE model whose encoder returns fixed-shape tensors."""
    B, T, C = feature_shape[0], feature_shape[1], feature_shape[2]
    tokens = torch.zeros(B, T, C)

    mock_encoder = MagicMock()
    mock_encoder.return_value = (tokens, None, None, None, None, None)

    mock_model = MagicMock()
    mock_model.model.encoder = mock_encoder
    return mock_model


class TestRunBatch:
    def test_output_shape(self) -> None:
        device = torch.device("cpu")
        n = 2
        chunks = [np.random.rand(64, 64, 64).astype(np.float32) for _ in range(n)]

        # Encoder returns (B, n_tokens + n_skip, C); spatial reshape expects 32^3 tokens after skip
        n_skip = 2
        n_spatial_tokens = 32 * 32 * 32
        C = 8
        mock_model = _make_mock_model((n, n_spatial_tokens + n_skip, C))

        out = run_batch(chunks, mock_model, device, n_skip_tokens=n_skip)
        assert out.shape == (n, C, 32, 32, 32)

    def test_nan_free_output(self) -> None:
        device = torch.device("cpu")
        chunks = [np.zeros((16, 16, 16), dtype=np.float32)]
        n_skip = 2
        mock_model = _make_mock_model((1, 32 * 32 * 32 + n_skip, 4))
        out = run_batch(chunks, mock_model, device, n_skip_tokens=n_skip)
        assert not torch.isnan(out).any()


class TestExtractFeatureVectorsTorch:
    def _feature_map(self, n_channels: int = 8) -> torch.Tensor:
        return torch.rand(n_channels, 16, 16, 16)

    def test_basic_extraction(self) -> None:
        fm = self._feature_map()
        centers = np.array([[64, 64, 64]])  # sub_volume_shape = (128, 128, 128)
        feats, mask = extract_feature_vectors_torch(fm, centers, (128, 128, 128), jump=8)
        assert feats.shape[1] == 8  # n_channels
        assert mask.shape == (1,)

    def test_out_of_bounds_centre_masked(self) -> None:
        fm = self._feature_map()
        centers = np.array([[9999, 9999, 9999]])
        feats, mask = extract_feature_vectors_torch(fm, centers, (128, 128, 128), jump=8)
        assert len(feats) == 0
        assert not mask[0]

    def test_return_mask_false(self) -> None:
        fm = self._feature_map()
        centers = np.array([[64, 64, 64]])
        result = extract_feature_vectors_torch(
            fm, centers, (128, 128, 128), jump=8, return_mask=False
        )
        assert isinstance(result, torch.Tensor)

    def test_multiple_centres(self) -> None:
        fm = self._feature_map(n_channels=4)
        centers = np.array([[32, 32, 32], [64, 64, 64], [96, 96, 96]])
        feats, mask = extract_feature_vectors_torch(fm, centers, (128, 128, 128), jump=4)
        assert mask.sum() == 3
        assert feats.shape == (3, 4)
