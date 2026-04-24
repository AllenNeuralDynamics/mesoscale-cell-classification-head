"""Tests for :mod:`mesoscale_cell_classification_head.spatial`."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cupy", reason="CuPy not available")
import cupy as cp  # noqa: E402

from mesoscale_cell_classification_head.spatial import (  # noqa: E402
    box_sum,
    build_hist_gpu,
    centered_slice,
    greedy_cover_gpu,
    integral_image,
    mask_coords_gpu,
)


class TestCenteredSlice:
    def test_basic(self) -> None:
        s = centered_slice(center=10, size=4, max_dim=20)
        assert s.start == 8
        assert s.stop == 12

    def test_clamp_at_start(self) -> None:
        s = centered_slice(center=1, size=6, max_dim=20)
        assert s.start == 0

    def test_clamp_at_end(self) -> None:
        s = centered_slice(center=18, size=6, max_dim=20)
        assert s.stop == 20

    def test_length_preserved(self) -> None:
        for center in [0, 5, 10, 19]:
            s = centered_slice(center=center, size=4, max_dim=20)
            assert s.stop - s.start == 4


class TestBuildHistGpu:
    def test_single_point(self) -> None:
        bins = cp.array([[1, 2, 3]], dtype=cp.int64)
        grid_shape = (5, 5, 5)
        hist = build_hist_gpu(bins, grid_shape)
        assert hist.shape == grid_shape
        assert int(hist[1, 2, 3]) == 1
        assert int(hist.sum()) == 1

    def test_multiple_points_same_bin(self) -> None:
        bins = cp.array([[0, 0, 0], [0, 0, 0]], dtype=cp.int64)
        hist = build_hist_gpu(bins, (3, 3, 3))
        assert int(hist[0, 0, 0]) == 2


class TestIntegralImage:
    def test_cumsum_correctness(self) -> None:
        hist = cp.zeros((4, 4, 4), dtype=cp.int64)
        hist[0, 0, 0] = 1
        hist[3, 3, 3] = 1
        integral = integral_image(hist)
        assert int(integral[3, 3, 3]) == 2
        assert int(integral[0, 0, 0]) == 1


class TestBoxSum:
    def test_full_volume(self) -> None:
        hist = cp.ones((4, 4, 4), dtype=cp.int64)
        integral = integral_image(hist)
        start = cp.array([0, 0, 0])
        end = cp.array([4, 4, 4])
        assert int(box_sum(integral, start, end)) == 64

    def test_single_voxel(self) -> None:
        hist = cp.zeros((4, 4, 4), dtype=cp.int64)
        hist[2, 2, 2] = 7
        integral = integral_image(hist)
        assert int(box_sum(integral, cp.array([2, 2, 2]), cp.array([3, 3, 3]))) == 7

    def test_partial_box(self) -> None:
        hist = cp.ones((4, 4, 4), dtype=cp.int64)
        integral = integral_image(hist)
        assert int(box_sum(integral, cp.array([0, 0, 0]), cp.array([2, 2, 2]))) == 8


class TestMaskCoordsGpu:
    def test_inside(self) -> None:
        coords = cp.array([[5, 5, 5]], dtype=cp.int64)
        mask = mask_coords_gpu(coords, cp.array([0, 0, 0]), cp.array([10, 10, 10]))
        assert bool(mask[0])

    def test_outside(self) -> None:
        coords = cp.array([[15, 5, 5]], dtype=cp.int64)
        mask = mask_coords_gpu(coords, cp.array([0, 0, 0]), cp.array([10, 10, 10]))
        assert not bool(mask[0])

    def test_boundary_exclusive(self) -> None:
        coords = cp.array([[10, 5, 5]], dtype=cp.int64)
        mask = mask_coords_gpu(coords, cp.array([0, 0, 0]), cp.array([10, 10, 10]))
        assert not bool(mask[0])


class TestGreedyCoverGpu:
    def test_all_points_covered(self) -> None:
        rng = np.random.default_rng(0)
        coords = rng.integers(0, 256, size=(50, 3)).astype(np.int64)
        shape = (256, 256, 256)
        box_size = cp.array([64, 64, 64], dtype=cp.int64)
        boxes, box_cells = greedy_cover_gpu(coords, shape, box_size, vox=16, verbose=False)

        covered = np.concatenate(box_cells) if box_cells else np.array([])
        assert len(np.unique(covered)) == len(coords)

    def test_empty_input(self) -> None:
        coords = np.empty((0, 3), dtype=np.int64)
        shape = (64, 64, 64)
        box_size = cp.array([32, 32, 32], dtype=cp.int64)
        boxes, box_cells = greedy_cover_gpu(coords, shape, box_size, vox=8, verbose=False)
        assert boxes == []
        assert box_cells == []
