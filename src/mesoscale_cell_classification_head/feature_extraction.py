"""MAE-based feature extraction for 3-D microscopy volumes.

Functions in this module accept raw numpy/zarr chunks, normalise them,
forward them through the Lightsheet3DMAE encoder, and return per-cell
feature vectors ready for downstream clustering.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def pad_to_shape(
    arr: np.ndarray,
    target_shape: tuple[int, int, int] = (128, 128, 128),
    mode: str = "constant",
    constant_values: float = 0,
) -> np.ndarray:
    """Pad a 3-D array to ``target_shape`` along each axis.

    Padding is applied only at the end (right side) of each axis so that
    the origin of ``arr`` aligns with the origin of the output.

    Parameters
    ----------
    arr : np.ndarray
        Input array with ``ndim == 3``.
    target_shape : tuple[int, int, int]
        Desired output shape ``(Z, Y, X)``.
    mode : str
        Padding mode forwarded to :func:`numpy.pad`.
    constant_values : float
        Fill value used when ``mode="constant"``.

    Returns
    -------
    np.ndarray
        Padded array with shape ``target_shape``.

    Raises
    ------
    AssertionError
        If ``arr.ndim != 3``.
    ValueError
        If any dimension of ``arr`` exceeds the corresponding target.
    """
    assert arr.ndim == 3, f"Expected 3D array, got shape {arr.shape}"

    pad_width = []
    for dim, target in zip(arr.shape, target_shape):
        if dim > target:
            raise ValueError(f"Array dimension {dim} exceeds target {target}")
        pad_width.append((0, target - dim))

    return np.pad(arr, pad_width=pad_width, mode=mode, constant_values=constant_values)


def run_batch(
    batch_chunks: list[np.ndarray],
    reconstruction_model: object,
    device: torch.device,
    n_skip_tokens: int = 2,
    array_shape: tuple[int, int, int] = (128, 128, 128),
    patch_shape: tuple[int, int, int] = (4, 4, 4),
    preprocessing_func: torch.nn.Module | None = None,
) -> torch.Tensor:
    """Forward a list of 3-D image chunks through the MAE encoder.

    Each chunk is padded to ``(128, 128, 128)``, stacked into a batch,
    percentile-clipped and z-score normalised, then passed through the
    encoder with ``mask_ratio=0``.  The returned tokens are reshaped back
    into a spatial feature map of shape ``(B, C, Dz, Dy, Dx)`` where
    ``Dz, Dy, Dx`` are determined by ``array_shape`` and ``patch_shape``.

    Parameters
    ----------
    batch_chunks : list[np.ndarray]
        List of ``(D, H, W)`` float32 numpy arrays.  All arrays are padded
        to ``(128, 128, 128)`` before batching.
    reconstruction_model : object
        A :class:`Lightsheet3DMAE` lightning module already moved to
        ``device`` and set to eval mode.
    device : torch.device
        Target compute device.
    n_skip_tokens : int
        Number of leading tokens to discard from the encoder output
        (e.g. CLS/register tokens).
    array_shape : tuple[int, int, int]
        Shape ``(D, H, W)`` to which each input chunk is padded.
    patch_shape : tuple[int, int, int]
        Shape ``(Pd, Ph, Pw)`` of the patches used by the MAE encoder;
        used to compute the spatial dimensions of the output feature map.

    Returns
    -------
    torch.Tensor
        Feature maps of shape ``(B, C, Dz, Dy, Dx)`` on ``device``,
        in float16 precision.
    """
    voxels_per_patch = tuple(
        array_shape[idx] // patch_shape[idx]
        for idx in range(len(array_shape))
    )

    padded = [
        pad_to_shape(arr, target_shape=array_shape, mode="constant", constant_values=0)
        for arr in batch_chunks
    ]
    stacked = np.stack(padded)
    batch = torch.from_numpy(stacked).unsqueeze(1).to(device).half()

    if preprocessing_func is not None:
        batch = preprocessing_func(batch)

    with torch.no_grad():
        feature_image, _, _, _, _, _ = reconstruction_model.model.encoder(
            batch.half(), mask_ratio=0.0, recover_layers=()
        )
        feature_image = feature_image[:, n_skip_tokens:, :]
        feature_image = torch.reshape(
            feature_image, (feature_image.shape[0], *voxels_per_patch, -1)
        )
        feature_image = torch.moveaxis(feature_image, -1, 1)

    return feature_image


def extract_feature_vectors_torch(
    feature_map: torch.Tensor,
    roi_centers_zyx: np.ndarray | torch.Tensor,
    jump: int,
    return_mask: bool = True,
    pool: bool = True,
) -> tuple[torch.Tensor, np.ndarray] | torch.Tensor:
    """Extract per-cell feature vectors from a spatial feature map.

    Parameters
    ----------
    feature_map : torch.Tensor
        ``(C, Dz, Dy, Dx)`` feature map on GPU.
    roi_centers_zyx : np.ndarray or torch.Tensor
        ``(N, 3)`` array of cell centre coordinates in the sub-volume frame
        ``(z, y, x)``.
    jump : int
        Desired local neighbourhood radius in image voxels; converted to
        feature-map voxels using the downsampling factors.  Ignored when
        ``pool=False``.
    return_mask : bool
        When ``True`` return a ``(feats, valid_mask)`` tuple where
        ``valid_mask`` flags which of the input centres fell inside the map.
    pool : bool
        When ``True`` (default) apply ``avg_pool3d`` over a ``jump``-radius
        neighbourhood before reading off features — smooths positional jitter,
        better for unsupervised clustering.  When ``False`` read the raw
        patch-level feature directly — preserves spatial detail, better for
        supervised MLP classification.

    Returns
    -------
    feats : torch.Tensor
        ``(N_valid, C)`` feature matrix on the same device as
        ``feature_map``.
    valid_mask : np.ndarray
        Boolean array of shape ``(N,)`` indicating valid centres.
        Only returned when ``return_mask=True``.
    """
    device = feature_map.device
    C, Dz, Dy, Dx = feature_map.shape

    # Padding in run_batch is appended at the END, so voxel 0 in the original
    # chunk maps to voxel 0 in the padded 128³ encoder input.  Patch i always
    # covers image voxels [i*p, i*p+p) regardless of whether the chunk is
    # smaller than 128 at a volume edge.
    _enc = 128  # run_batch always pads to this size
    fz, fy, fx = _enc / Dz, _enc / Dy, _enc / Dx

    centers = torch.as_tensor(roi_centers_zyx, device=device, dtype=torch.float32)
    cz = torch.round(centers[:, 0] / fz).long()
    cy = torch.round(centers[:, 1] / fy).long()
    cx = torch.round(centers[:, 2] / fx).long()

    valid = (
        (cz >= 0) & (cz < Dz) & (cy >= 0) & (cy < Dy) & (cx >= 0) & (cx < Dx)
    )

    if not valid.any():
        empty: torch.Tensor = torch.empty(0, C, device=device)
        return (empty, valid.cpu().numpy()) if return_mask else empty

    cz, cy, cx = cz[valid], cy[valid], cx[valid]

    if pool:
        size_z = min(max(1, int(math.ceil(jump / fz))), Dz)
        size_y = min(max(1, int(math.ceil(jump / fy))), Dy)
        size_x = min(max(1, int(math.ceil(jump / fx))), Dx)
        source = F.avg_pool3d(
            feature_map.unsqueeze(0),
            kernel_size=(size_z, size_y, size_x),
            stride=1,
            padding=0,
        ).squeeze(0)
    else:
        source = feature_map

    cz = torch.clamp(cz, 0, source.shape[1] - 1)
    cy = torch.clamp(cy, 0, source.shape[2] - 1)
    cx = torch.clamp(cx, 0, source.shape[3] - 1)

    feats = source[:, cz, cy, cx].T  # (N_valid, C)

    return (feats, valid.cpu().numpy()) if return_mask else feats
