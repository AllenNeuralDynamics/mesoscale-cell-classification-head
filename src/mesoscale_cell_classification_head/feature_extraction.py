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
    remove_cls_token: bool = True,
    remove_register_tokens: bool = True,
    array_shape: tuple[int, int, int] = (128, 128, 128),
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
    remove_cls_token : bool
        Whether to remove the CLS token from the encoder output.
    array_shape : tuple[int, int, int]
        Shape ``(D, H, W)`` to which each input chunk is padded.

    Returns
    -------
    torch.Tensor
        Feature maps of shape ``(B, C, Dz, Dy, Dx)`` on ``device``,
        in float16 precision.
    """
    n_cls = 1
    n_register_tokens = reconstruction_model.model.encoder.n_register_tokens

    with torch.inference_mode():
        batch = torch.stack([
            torch.as_tensor(
                pad_to_shape(arr, array_shape, mode="constant", constant_values=0)
            )
            for arr in batch_chunks
        ], dim=0).unsqueeze(1).to(device, dtype=torch.float16)

        if preprocessing_func is not None:
            batch = preprocessing_func(batch)

        feature_image, _, _, _, _, _ = reconstruction_model.model.encoder(
            batch.half(), mask_ratio=0.0, recover_layers=()
        )

        start = 0
        parts = []

        if not remove_cls_token:
            parts.append(feature_image[:, start:start+n_cls, :])
        start += n_cls

        if not remove_register_tokens:
            parts.append(feature_image[:, start:start+n_register_tokens, :])
        start += n_register_tokens

        parts.append(feature_image[:, start:, :])

        feature_image = torch.cat(parts, dim=1)

        # Reshape pure patch-token output to spatial feature map (B, C, Dg, Hg, Wg).
        # Only valid when CLS and register tokens are removed (the default), leaving
        # exactly Dg*Hg*Wg tokens that map one-to-one onto the spatial grid.
        B, N_tokens, C_out = feature_image.shape
        Dg, Hg, Wg = reconstruction_model.model.encoder.grid_size
        if N_tokens == Dg * Hg * Wg:
            feature_image = (
                feature_image.reshape(B, Dg, Hg, Wg, C_out)
                .permute(0, 4, 1, 2, 3)
                .contiguous()
            )

    return feature_image


def extract_feature_vectors_torch(
    feature_map: torch.Tensor,
    roi_centers_zyx: np.ndarray | torch.Tensor,
    return_mask: bool = True,
    pad_size: int = 128,
) -> tuple[torch.Tensor, np.ndarray] | torch.Tensor:
    """Extract per-cell feature vectors from a spatial feature map.

    Maps each cell centre to the nearest patch token and returns that token's
    embedding.  Each MAE patch token already encodes the full local voxel
    neighbourhood (patch_size³), making this the most discriminative
    per-cell representation for a flat MLP classifier.

    Parameters
    ----------
    feature_map : torch.Tensor
        ``(C, Dz, Dy, Dx)`` feature map, channels-first, on GPU.
        This is the direct output of :func:`run_batch` indexed at batch dim.
    roi_centers_zyx : np.ndarray or torch.Tensor
        ``(N, 3)`` cell centre coordinates in the sub-volume frame ``(z, y, x)``.
    return_mask : bool
        When ``True`` return ``(feats, valid_mask)`` where ``valid_mask``
        flags centres that fell inside the map.
    pad_size : int
        Size to which each chunk was padded by :func:`run_batch`; used to
        convert image-space coordinates to patch-space indices.

    Returns
    -------
    feats : torch.Tensor
        ``(N_valid, C)`` feature matrix on the same device as ``feature_map``.
    valid_mask : np.ndarray
        Boolean ``(N,)`` array. Only returned when ``return_mask=True``.
    """
    device = feature_map.device
    C, Dz, Dy, Dx = feature_map.shape

    fz, fy, fx = pad_size / Dz, pad_size / Dy, pad_size / Dx

    centers = torch.as_tensor(roi_centers_zyx, device=device, dtype=torch.float32)
    cz = torch.round(centers[:, 0] / fz).long()
    cy = torch.round(centers[:, 1] / fy).long()
    cx = torch.round(centers[:, 2] / fx).long()

    valid = (
        (cz >= 0) & (cz < Dz) &
        (cy >= 0) & (cy < Dy) &
        (cx >= 0) & (cx < Dx)
    )

    if not valid.any():
        empty = torch.empty(0, C, device=device)
        return (empty, valid.cpu().numpy()) if return_mask else empty

    cz = torch.clamp(cz[valid], 0, Dz - 1)
    cy = torch.clamp(cy[valid], 0, Dy - 1)
    cx = torch.clamp(cx[valid], 0, Dx - 1)

    feats = feature_map[:, cz, cy, cx].T  # (N_valid, C)
    return (feats, valid.cpu().numpy()) if return_mask else feats


def extract_feature_vectors_torch_3d(
    feature_map: torch.Tensor,
    roi_centers_zyx: np.ndarray | torch.Tensor,
    return_mask: bool = True,
    cube_size: int = 6,
    pad_size: int = 128,
) -> tuple[torch.Tensor, np.ndarray] | torch.Tensor:
    """Extract a spatial patch cube around each cell centre.

    Returns a ``(K, K, K, C)`` cube of patch tokens per cell, useful for
    visualisation (PCA, attention maps) or a 3-D CNN classifier head.
    For a flat MLP, prefer :func:`extract_feature_vectors_torch` instead.

    Parameters
    ----------
    feature_map : torch.Tensor
        ``(Dz, Dy, Dx, C)`` feature map, channels-last.
        Build it from the :func:`run_batch` output via
        ``fm.permute(1, 2, 3, 0)`` or the reshape path in
        :func:`_iter_cell_features` with ``use_3d_features=True``.
    roi_centers_zyx : np.ndarray or torch.Tensor
        ``(N, 3)`` cell centre coordinates in the sub-volume frame ``(z, y, x)``.
    return_mask : bool
        When ``True`` return ``(feats, valid_mask)``.
    cube_size : int
        Side length of the extracted patch cube in patch units.
    pad_size : int
        Size to which each chunk was padded by :func:`run_batch`.

    Returns
    -------
    feats : torch.Tensor
        ``(N_valid, K, K, K, C)`` patch cubes on the same device.
    valid_mask : np.ndarray
        Boolean ``(N,)`` array. Only returned when ``return_mask=True``.
    """
    device = feature_map.device
    Dz, Dy, Dx, C = feature_map.shape
    radius = cube_size // 2

    fz, fy, fx = pad_size / Dz, pad_size / Dy, pad_size / Dx

    centers = torch.as_tensor(roi_centers_zyx, device=device, dtype=torch.float32)

    cz = torch.round(centers[:, 0] / fz).long()
    cy = torch.round(centers[:, 1] / fy).long()
    cx = torch.round(centers[:, 2] / fx).long()

    valid = (
        (cz >= 0) & (cz < Dz) &
        (cy >= 0) & (cy < Dy) &
        (cx >= 0) & (cx < Dx)
    )

    if not valid.any():
        empty = torch.empty(0, cube_size, cube_size, cube_size, C, device=device)
        return (empty, valid.cpu().numpy()) if return_mask else empty

    cz, cy, cx = cz[valid], cy[valid], cx[valid]

    # Pad the feature map so cubes centred near the boundary stay in-bounds
    feature_map_pad = F.pad(
        feature_map.permute(3, 0, 1, 2),          # (C, Dz, Dy, Dx) for F.pad
        (radius, radius, radius, radius, radius, radius),
        mode="constant",
        value=0,
    ).permute(1, 2, 3, 0)                          # back to (Dz+2r, Dy+2r, Dx+2r, C)

    cz = cz + radius
    cy = cy + radius
    cx = cx + radius

    offsets = torch.arange(-radius, radius, device=device)
    zz, yy, xx = torch.meshgrid(offsets, offsets, offsets, indexing="ij")

    iz = (cz[:, None, None, None] + zz[None]).clamp(0, feature_map_pad.shape[0] - 1)
    iy = (cy[:, None, None, None] + yy[None]).clamp(0, feature_map_pad.shape[1] - 1)
    ix = (cx[:, None, None, None] + xx[None]).clamp(0, feature_map_pad.shape[2] - 1)
    subvols = feature_map_pad[iz, iy, ix, :]  # (N, K, K, K, C)
    return (subvols, valid.cpu().numpy()) if return_mask else subvols
