"""GPU-accelerated spatial indexing and greedy box-cover utilities.

All heavy computation runs on GPU via CuPy; CPU-side numpy arrays are
accepted at function boundaries and converted internally.
"""

from __future__ import annotations

import cupy as cp
import numpy as np
from tqdm import tqdm


def centered_slice(center: int, size: int, max_dim: int) -> slice:
    """Return a slice of length ``size`` centred on ``center``.

    The slice is clamped so it never exceeds ``[0, max_dim)``.

    Parameters
    ----------
    center : int
        Desired centre index.
    size : int
        Length of the slice.
    max_dim : int
        Upper bound (exclusive) for the slice end.

    Returns
    -------
    slice
        A valid ``slice(start, end)`` with ``end - start == size``
        wherever the volume allows.
    """
    half = size // 2
    start = int(max(0, center - half))
    end = int(min(max_dim, start + size))
    start = int(max(0, end - size))
    return slice(start, end)


def build_hist_gpu(bins: cp.ndarray, grid_shape: tuple[int, int, int]) -> cp.ndarray:
    """Build a 3-D occupancy histogram on the GPU.

    Parameters
    ----------
    bins : cp.ndarray
        ``(N, 3)`` CuPy integer array of voxel coordinates ``(Z, Y, X)``.
    grid_shape : tuple[int, int, int]
        Shape ``(GZ, GY, GX)`` of the output histogram.

    Returns
    -------
    cp.ndarray
        3-D histogram of shape ``grid_shape`` on the GPU.
    """
    flat_idx = (
        bins[:, 0] * (grid_shape[1] * grid_shape[2])
        + bins[:, 1] * grid_shape[2]
        + bins[:, 2]
    )
    counts = cp.bincount(
        flat_idx, minlength=grid_shape[0] * grid_shape[1] * grid_shape[2]
    )
    return counts.reshape(grid_shape)


def integral_image(hist: cp.ndarray) -> cp.ndarray:
    """Compute the 3-D prefix-sum (integral image) of ``hist``.

    Parameters
    ----------
    hist : cp.ndarray
        3-D occupancy histogram on the GPU.

    Returns
    -------
    cp.ndarray
        Cumulative sum along all three axes, same shape as ``hist``.
    """
    return hist.cumsum(axis=0).cumsum(axis=1).cumsum(axis=2)


def box_sum(
    integral: cp.ndarray,
    start: cp.ndarray,
    end: cp.ndarray,
) -> cp.ndarray:
    """Sum the values inside an axis-aligned box using an integral image.

    Parameters
    ----------
    integral : cp.ndarray
        3-D integral image (output of :func:`integral_image`).
    start : cp.ndarray
        1-D array ``[z0, y0, x0]`` — inclusive lower corner (voxel indices).
    end : cp.ndarray
        1-D array ``[z1, y1, x1]`` — exclusive upper corner (voxel indices).

    Returns
    -------
    cp.ndarray
        Scalar CuPy array with the box count.
    """
    z0, y0, x0 = start
    z1, y1, x1 = end - 1  # convert to inclusive

    total = integral[z1, y1, x1]
    total -= integral[z0 - 1, y1, x1] if z0 > 0 else 0
    total -= integral[z1, y0 - 1, x1] if y0 > 0 else 0
    total -= integral[z1, y1, x0 - 1] if x0 > 0 else 0
    total += integral[z0 - 1, y0 - 1, x1] if z0 > 0 and y0 > 0 else 0
    total += integral[z0 - 1, y1, x0 - 1] if z0 > 0 and x0 > 0 else 0
    total += integral[z1, y0 - 1, x0 - 1] if y0 > 0 and x0 > 0 else 0
    total -= integral[z0 - 1, y0 - 1, x0 - 1] if z0 > 0 and y0 > 0 and x0 > 0 else 0
    return total


def mask_coords_gpu(
    coords: cp.ndarray,
    start: cp.ndarray,
    end: cp.ndarray,
) -> cp.ndarray:
    """Return a boolean mask for coordinates inside ``[start, end)``.

    Parameters
    ----------
    coords : cp.ndarray
        ``(N, 3)`` CuPy integer array of world coordinates ``(Z, Y, X)``.
    start : cp.ndarray
        1-D array ``[z0, y0, x0]`` — inclusive lower bound.
    end : cp.ndarray
        1-D array ``[z1, y1, x1]`` — exclusive upper bound.

    Returns
    -------
    cp.ndarray
        Boolean mask of shape ``(N,)`` on the GPU.
    """
    return cp.all((coords >= start) & (coords < end), axis=1)


def greedy_cover_gpu(
    coords_cpu: np.ndarray,
    shape: tuple[int, ...],
    box_size: cp.ndarray,
    vox: int,
    verbose: bool = True,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[np.ndarray]]:
    """Cover a point cloud with boxes using a greedy density-first strategy.

    The algorithm repeatedly places an axis-aligned box centred on the
    densest occupied voxel and removes the covered points until no points
    remain. All histogram operations run on the GPU via CuPy.

    Parameters
    ----------
    coords_cpu : np.ndarray
        ``(N, 3)`` NumPy array of point coordinates ``(Z, Y, X)``.
    shape : tuple[int, ...]
        Bounding-box shape of the full region ``(D, H, W)``.
    box_size : cp.ndarray
        1-D CuPy array ``[bz, by, bx]`` — box side lengths in world units.
    vox : int
        Voxelisation factor: coordinates are divided by ``vox`` before
        histogram construction to reduce memory usage.
    verbose : bool
        Print summary statistics when ``True``.

    Returns
    -------
    boxes : list[tuple[np.ndarray, np.ndarray]]
        List of ``(start, end)`` pairs in world coordinates (NumPy arrays).
    box_cells : list[np.ndarray]
        Parallel list of 1-D index arrays into ``coords_cpu`` for each box.
    """
    coords = cp.asarray(coords_cpu, dtype=cp.int64)
    bins = coords // vox

    grid_shape = (
        shape[0] // vox + 1,
        shape[1] // vox + 1,
        shape[2] // vox + 1,
    )

    box_vox = box_size // vox
    grid_cp = cp.array(list(grid_shape), dtype=cp.int64)

    # Build histogram once; update incrementally instead of rebuilding each iter.
    hist = build_hist_gpu(bins, grid_shape)
    # Track remaining points as a shrinking index array (avoids cp.where each iter).
    active_indices = cp.arange(coords.shape[0], dtype=cp.int64)

    boxes: list[tuple[np.ndarray, np.ndarray]] = []
    box_cells: list[np.ndarray] = []

    if verbose:
        print(f"greedy_cover_gpu: {len(active_indices)} points, "
              f"grid_shape={grid_shape}, box_vox={box_vox.tolist()}, "
              f"hist_sum={int(hist.sum())}, hist_max={int(hist.max())}")

    pbar = tqdm(total=int(coords.shape[0]), desc="Covered points")

    while len(active_indices) > 0:
        if int(hist.max()) == 0:
            if verbose:
                print(f"No more dense boxes found — "
                      f"{len(active_indices)} points remain uncovered.")
            break

        best_voxel_idx = cp.argmax(hist)
        best_voxel = cp.stack(cp.unravel_index(best_voxel_idx, hist.shape))
        best_start_vox = cp.clip(best_voxel - box_vox // 2, 0, grid_cp - box_vox)

        start = best_start_vox * vox
        end = start + box_size

        mask = mask_coords_gpu(coords[active_indices], start, end)
        covered = int(cp.sum(mask))
        if covered == 0:
            if verbose:
                print(f"Warning: densest voxel {best_voxel.tolist()} "
                      f"(hist={int(hist.max())}) produced a box "
                      f"[{start.tolist()}, {end.tolist()}) covering 0 points. "
                      f"coords range: min={coords[active_indices].min(axis=0).tolist()}, "
                      f"max={coords[active_indices].max(axis=0).tolist()}")
            break

        covered_global = active_indices[mask]
        box_cells.append(covered_global.get())
        boxes.append((cp.asnumpy(start), cp.asnumpy(end)))

        # Subtract covered points from histogram in-place (O(covered), not O(N)).
        covered_bins = bins[covered_global]
        flat_idx = (
            covered_bins[:, 0] * (grid_shape[1] * grid_shape[2])
            + covered_bins[:, 1] * grid_shape[2]
            + covered_bins[:, 2]
        )
        hist -= cp.bincount(flat_idx, minlength=hist.size).reshape(grid_shape).astype(hist.dtype)

        active_indices = active_indices[~mask]
        pbar.update(covered)

    pbar.close()

    if verbose:
        print("Greedy boxes:", len(boxes))
        if box_cells:
            print("Avg cells per box:", np.mean([len(b) for b in box_cells]))
        else:
            print("Avg cells per box: nan")

    return boxes, box_cells
