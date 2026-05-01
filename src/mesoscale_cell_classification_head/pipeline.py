"""End-to-end cell-classification pipeline.

The pipeline has three phases:

1. Data loading — cell proposals from S3 CSV, 3-D image from S3 Zarr.
2. Training — greedy box cover -> feature extraction -> incremental PCA
   -> online k-means EMA updates.
3. Inference — frozen PCA + k-means assigns a label to every proposed
   cell; results saved as a compressed .npz file.
"""

from __future__ import annotations

import argparse
import logging
import warnings
from collections.abc import Iterator
from functools import partial
from typing import Callable

import cupy as cp
import dask.array as da
import numpy as np
import pandas as pd
import s3fs
import torch
from aind_lightsheet_mae.model.lightning_modules import Lightsheet3DMAE
from sklearn.decomposition import IncrementalPCA
from tqdm import tqdm

from mesoscale_cell_classification_head.clustering import OnlineKMeans
from mesoscale_cell_classification_head.feature_extraction import (
    extract_feature_vectors_torch,
    run_batch,
)
from mesoscale_cell_classification_head.spatial import greedy_cover_gpu

from .preprocessing import apply_transform, zscore_val_augmentations

logger = logging.getLogger(__name__)

def _default_config(
    box_dim: int = 128,
    patch_size: int = 4,
    overlap: int = 10,
) -> dict:
    box_core = box_dim - overlap
    vox = box_dim // patch_size
    return {
        # Greedy cover uses the core box.  Each loaded chunk is box_dim cubic, centred
        # on the core, so border cells have halo = overlap//2 voxels of real data
        # context on every face instead of zero-padding.
        "box_size": cp.array([box_core] * 3, dtype=cp.int64),
        "box_dim": box_dim,
        "overlap": overlap,
        "patch_size": patch_size,
        "vox": vox,
        "scale": 1,
        "max_boxes": 30000,
        "batch_size": 4,
        "jump": 8,
        "n_clusters": 4,
        "n_pca_components": 64,
        "max_cells_per_box": 128,
        "ipca_warmup_boxes": 200,
        "store_kmeans_every_n": 500,
        "lr_kmeans": 0.03,
        "head_batch_size": 1024,
    }

def load_data(
    s3_proposals_path: str,
    dataset_path: str,
    scale: int,
) -> tuple[np.ndarray, da.Array]:
    """Load cell proposals and the image Zarr from S3.

    Parameters
    ----------
    s3_proposals_path : str
        S3 URI of the CSV file containing z, y, x columns.
    dataset_path : str
        S3 URI of the OME-Zarr dataset root.
    scale : int
        Zarr resolution level to use.  Coordinates are downsampled by
        2**scale.

    Returns
    -------
    cell_zyx_ds : np.ndarray
        (N, 3) int64 array of downsampled cell coordinates.
    loaded_zarr : da.Array
        Dask array for the chosen Zarr resolution level.
    """
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(s3_proposals_path, mode="rb") as f:
        proposals_df = pd.read_csv(f)

    cell_zyx = proposals_df[["z", "y", "x"]].values.astype(np.uint32)
    logger.info("Proposed cell coordinates shape: %s", cell_zyx.shape)

    loaded_zarr = da.squeeze(da.from_zarr(f"{dataset_path}/{scale}"))
    cell_zyx_ds = (cell_zyx // (2**scale)).astype(np.int64)
    logger.info("Downsampled coordinates shape: %s  zarr shape: %s",
                cell_zyx_ds.shape, loaded_zarr.shape)

    return cell_zyx_ds, loaded_zarr

def _load_box_batch(
    batch_indices: list[int],
    boxes: list[tuple[np.ndarray, np.ndarray]],
    box_cells_ids: list[np.ndarray],
    cell_zyx_ds: np.ndarray,
    loaded_zarr: da.Array,
    overlap: int = 0,
    box_dim: int = 128,
) -> tuple[list[np.ndarray], list[np.ndarray], list[tuple[int, np.ndarray]]]:
    """Load image chunks and local cell coordinates for a batch of boxes.

    Each chunk is a box_dim³ window centred on the assignment box (core).
    The overlap // 2 voxels on each face are real image data from the
    neighbouring box's territory, giving border cells a complete encoder
    context.  At volume edges the window is clamped to [0, zarr_shape)
    and pad_to_shape (inside :func:`run_batch`) fills the shortfall with
    zeros.

    Parameters
    ----------
    batch_indices : list[int]
        Indices into boxes / box_cells_ids to load.
    boxes : list[tuple[np.ndarray, np.ndarray]]
        List of (start, end) world-coordinate arrays for the core boxes.
    box_cells_ids : list[np.ndarray]
        Parallel list of cell-index arrays (into cell_zyx_ds).
    cell_zyx_ds : np.ndarray
        Full (N, 3) downsampled coordinate array.
    loaded_zarr : da.Array
        Dask array for the image.
    overlap : int
        Total voxels of halo per axis.  halo_per_face = overlap // 2.
        When 0 (default) the window is exactly the core box — no change
        in behaviour.
    box_dim : int
        Side length of the loaded chunk, i.e. the encoder input size.
        Must equal box_core + overlap.

    Returns
    -------
    batch_chunks : list[np.ndarray]
        Image sub-volumes of shape ≤ (box_dim, box_dim, box_dim), one
        per non-empty box.
    batch_points : list[np.ndarray]
        Cell coordinates in the **padded chunk frame**, parallel to
        batch_chunks.
    batch_starts : list[tuple[int, np.ndarray]]
        (original_box_index, padded_start_zyx) pairs.  Adding
        padded_start to the local coordinates recovers world coordinates.
    """
    halo = overlap // 2

    batch_chunks: list[np.ndarray] = []
    batch_points: list[np.ndarray] = []
    batch_starts: list[tuple[int, np.ndarray]] = []

    for i in batch_indices:
        start, _ = boxes[i]
        z0, y0, x0 = map(int, start)

        curr_box_points = cell_zyx_ds[box_cells_ids[i]]
        if len(curr_box_points) == 0:
            continue

        pz0 = max(0, z0 - halo)
        py0 = max(0, y0 - halo)
        px0 = max(0, x0 - halo)

        chunk = np.asarray(
            loaded_zarr[
                pz0 : pz0 + box_dim,
                py0 : py0 + box_dim,
                px0 : px0 + box_dim,
            ].compute(),
            dtype=np.float32,
        )
        padded_start = np.array([pz0, py0, px0])
        batch_points.append(curr_box_points - padded_start)
        batch_starts.append((i, padded_start))
        batch_chunks.append(chunk)

    return batch_chunks, batch_points, batch_starts

def train(
    boxes: list[tuple[np.ndarray, np.ndarray]],
    box_cells_ids: list[np.ndarray],
    cell_zyx_ds: np.ndarray,
    loaded_zarr: da.Array,
    reconstruction_model: object,
    device: torch.device,
    cfg: dict,
    val_transform: Callable = None
) -> tuple[IncrementalPCA, OnlineKMeans | None]:
    """Fit IncrementalPCA and OnlineKMeans on box-extracted features.

    Parameters
    ----------
    boxes : list[tuple[np.ndarray, np.ndarray]]
        Ordered list of (start, end) box corners.
    box_cells_ids : list[np.ndarray]
        Parallel list mapping each box to cell indices in cell_zyx_ds.
    cell_zyx_ds : np.ndarray
        (N, 3) downsampled cell coordinates.
    loaded_zarr : da.Array
        Image data source.
    reconstruction_model : object
        Pretrained :class:`Lightsheet3DMAE` in eval mode.
    device : torch.device
        Compute device.
    cfg : dict
        Pipeline configuration (see :func:`_default_config`).
    val_transform : Callable, optional
        Optional transformation function applied to each batch chunk before encoding.
        Should take and return a numpy array of the same shape.

    Returns
    -------
    ipca : IncrementalPCA
        Fitted incremental PCA transformer.
    kmeans : OnlineKMeans or None
        Trained k-means, or None if fewer than
        cfg["ipca_warmup_boxes"] valid batches were seen.
    """
    ipca = IncrementalPCA(n_components=cfg["n_pca_components"], whiten=True)
    kmeans: OnlineKMeans | None = None
    preprocessing_func = None
    if val_transform is not None:
        preprocessing_func = partial(apply_transform, transform=val_transform)

    boxes_for_ipca = 0
    processed_boxes = 0

    n_boxes = min(cfg["max_boxes"], len(boxes))
    for batch_start in tqdm(
        range(0, n_boxes, cfg["batch_size"]), desc="Training pass"
    ):
        batch_indices = list(range(batch_start, min(batch_start + cfg["batch_size"], n_boxes)))
        batch_chunks, batch_points, batch_starts = _load_box_batch(
            batch_indices, boxes, box_cells_ids, cell_zyx_ds, loaded_zarr,
            overlap=cfg["overlap"], box_dim=cfg["box_dim"],
        )
        if not batch_chunks:
            continue

        feature_maps = run_batch(
            batch_chunks=batch_chunks,
            reconstruction_model=reconstruction_model,
            device=device,
            preprocessing_func=preprocessing_func,
        )
        feature_maps = torch.nan_to_num(
            feature_maps, nan=float(torch.finfo(feature_maps.dtype).tiny)
        )

        for bidx, fm in enumerate(feature_maps):
            feats, valid_mask = extract_feature_vectors_torch(
                feature_map=fm,
                roi_centers_zyx=batch_points[bidx],
                jump=cfg["jump"],
                return_mask=True,
            )
            if len(feats) == 0:
                continue

            feats = feats.to(device).float()
            global_box_id, start_offset = batch_starts[bidx]
            global_points = (batch_points[bidx] + start_offset)[valid_mask].astype(np.uint32)

            feats_capped, global_points_capped = _cap_features(
                feats, global_points, cfg["max_cells_per_box"], device
            )

            feats_np = feats_capped.cpu().numpy()
            if feats_np.shape[0] >= cfg["n_pca_components"]:
                ipca.partial_fit(feats_np)
                boxes_for_ipca += 1

            if boxes_for_ipca >= cfg["ipca_warmup_boxes"]:
                reduced = torch.from_numpy(ipca.transform(feats_np)).to(device).float()
                if kmeans is None:
                    kmeans = OnlineKMeans(
                        n_clusters=cfg["n_clusters"],
                        dim=cfg["n_pca_components"],
                        lr=cfg["lr_kmeans"],
                        device=str(device),
                    )
                kmeans.update(x=reduced, global_points=global_points_capped)

            processed_boxes += 1
            if processed_boxes % cfg["store_kmeans_every_n"] == 0 and kmeans is not None:
                torch.save(kmeans.state_dict(), f"kmeans_state_{processed_boxes}.pt")

            del feats

    if kmeans is not None:
        torch.save(kmeans.state_dict(), "kmeans_state_trained.pt")
        logger.info("Centroid counts after training: %s", kmeans.counts.cpu().numpy())

    return ipca, kmeans


def _cap_features(
    feats: torch.Tensor,
    global_points: np.ndarray,
    max_cells: int,
    device: torch.device,
) -> tuple[torch.Tensor, np.ndarray]:
    """Randomly subsample features to at most max_cells rows.

    Parameters
    ----------
    feats : torch.Tensor
        (N, C) feature matrix.
    global_points : np.ndarray
        (N, 3) world coordinates parallel to feats.
    max_cells : int
        Maximum number of rows to keep.
    device : torch.device
        Device for the random permutation.

    Returns
    -------
    feats_capped : torch.Tensor
        Sub-sampled feature matrix, at most (max_cells, C).
    global_points_capped : np.ndarray
        Corresponding sub-sampled coordinates.
    """
    n = feats.shape[0]
    if n <= max_cells:
        return feats, global_points
    idx = torch.randperm(n, device=device)[:max_cells]
    return feats[idx], global_points[idx.cpu().numpy()]

def _iter_cell_features(
    boxes: list[tuple[np.ndarray, np.ndarray]],
    box_cells_ids: list[np.ndarray],
    cell_zyx_ds: np.ndarray,
    loaded_zarr: da.Array,
    reconstruction_model: object,
    device: torch.device,
    cfg: dict,
    desc: str = "Feature extraction",
    val_transform: Callable = None,
    pool: bool = True,
) -> Iterator[tuple[torch.Tensor, np.ndarray]]:
    """Yield per-cell feature vectors and world coordinates for every box.

    Handles data loading, MAE encoding, and feature extraction in a single
    pass.  Callers receive one (feats, global_points) tuple per non-empty
    box and are responsible for any downstream classification or batching.

    Parameters
    ----------
    boxes : list[tuple[np.ndarray, np.ndarray]]
        Ordered list of (start, end) box corners.
    box_cells_ids : list[np.ndarray]
        Parallel list mapping each box to cell indices in cell_zyx_ds.
    cell_zyx_ds : np.ndarray
        (N, 3) downsampled cell coordinates.
    loaded_zarr : da.Array
        Image data source.
    reconstruction_model : object
        Pretrained :class:`Lightsheet3DMAE` in eval mode.
    device : torch.device
        Compute device.
    cfg : dict
        Pipeline configuration (see :func:`_default_config`).  Uses
        max_boxes, batch_size, overlap, box_dim, and jump.
    desc : str
        Label shown on the tqdm progress bar.
    val_transform : Callable, optional
        Optional transformation function applied to each batch chunk before encoding.
        Should take and return a numpy array of the same shape.

    Yields
    ------
    feats : torch.Tensor
        (N_valid, C) float32 feature matrix on device.
    global_points : np.ndarray
        (N_valid, 3) uint32 world coordinates parallel to feats.
    """
    preprocessing_func = None
    if val_transform is not None:
        preprocessing_func = partial(apply_transform, transform=val_transform)

    n_boxes = min(cfg["max_boxes"], len(boxes))
    for batch_start in tqdm(range(0, n_boxes, cfg["batch_size"]), desc=desc):
        batch_indices = list(range(batch_start, min(batch_start + cfg["batch_size"], n_boxes)))
        batch_chunks, batch_points, batch_starts = _load_box_batch(
            batch_indices, boxes, box_cells_ids, cell_zyx_ds, loaded_zarr,
            overlap=cfg["overlap"], box_dim=cfg["box_dim"],
        )
        if not batch_chunks:
            continue

        feature_maps = run_batch(
            batch_chunks=batch_chunks,
            reconstruction_model=reconstruction_model,
            device=device,
            preprocessing_func=preprocessing_func,
        )
        feature_maps = torch.nan_to_num(
            feature_maps, nan=float(torch.finfo(feature_maps.dtype).tiny)
        )

        for bidx, fm in enumerate(feature_maps):
            feats, valid_mask = extract_feature_vectors_torch(
                feature_map=fm,
                roi_centers_zyx=batch_points[bidx],
                jump=cfg["jump"],
                return_mask=True,
                pool=pool,
            )
            if len(feats) == 0:
                continue

            _, start_offset = batch_starts[bidx]
            global_points = (batch_points[bidx] + start_offset)[valid_mask].astype(np.uint32)
            yield feats.to(device).float(), global_points


def infer_clustering(
    boxes: list[tuple[np.ndarray, np.ndarray]],
    box_cells_ids: list[np.ndarray],
    cell_zyx_ds: np.ndarray,
    loaded_zarr: da.Array,
    reconstruction_model: object,
    device: torch.device,
    ipca: IncrementalPCA,
    kmeans: OnlineKMeans,
    cfg: dict,
    val_transform: Callable = None
) -> tuple[np.ndarray, np.ndarray]:
    """Assign cluster labels to all cells using frozen PCA and k-means.

    Parameters
    ----------
    boxes : list[tuple[np.ndarray, np.ndarray]]
        Box corners (same order as training).
    box_cells_ids : list[np.ndarray]
        Parallel cell-index lists.
    cell_zyx_ds : np.ndarray
        (N, 3) downsampled cell coordinates.
    loaded_zarr : da.Array
        Image data source.
    reconstruction_model : object
        Pretrained :class:`Lightsheet3DMAE` in eval mode.
    device : torch.device
        Compute device.
    ipca : IncrementalPCA
        Fitted PCA transformer (frozen — not updated here).
    kmeans : OnlineKMeans
        Trained k-means (predict-only — centroids are not updated here).
    cfg : dict
        Pipeline configuration (see :func:`_default_config`).
    val_transform : Callable, optional
        Optional transformation function applied to each batch chunk before encoding.
        Should take and return a numpy array of the same shape.

    Returns
    -------
    all_points : np.ndarray
        (M, 3) uint32 world coordinates of classified cells.
    all_labels : np.ndarray
        (M,) uint8 cluster labels parallel to all_points.
    """
    all_points: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for feats, global_points in _iter_cell_features(
        boxes, box_cells_ids, cell_zyx_ds, loaded_zarr,
        reconstruction_model, device, cfg, desc="Clustering inference",
        val_transform=val_transform,
    ):
        reduced = torch.from_numpy(
            ipca.transform(feats.cpu().numpy())
        ).to(device).float()
        labels = kmeans.predict(reduced)
        all_points.append(global_points)
        all_labels.append(labels.cpu().numpy().astype(np.uint8))

    if not all_points:
        return np.empty((0, 3), dtype=np.uint32), np.empty(0, dtype=np.uint8)

    return np.concatenate(all_points, axis=0), np.concatenate(all_labels, axis=0)


def infer_head(
    boxes: list[tuple[np.ndarray, np.ndarray]],
    box_cells_ids: list[np.ndarray],
    cell_zyx_ds: np.ndarray,
    loaded_zarr: da.Array,
    reconstruction_model: object,
    device: torch.device,
    head_model: torch.nn.Module,
    cfg: dict,
    val_transform: Callable = None
) -> tuple[np.ndarray, np.ndarray]:
    """Assign labels to all cells using a PyTorch head model.

    Features from multiple boxes are accumulated into a single batch before
    calling head_model, improving GPU utilisation compared to processing
    one box at a time.

    Parameters
    ----------
    boxes : list[tuple[np.ndarray, np.ndarray]]
        Box corners (same order as training).
    box_cells_ids : list[np.ndarray]
        Parallel cell-index lists.
    cell_zyx_ds : np.ndarray
        (N, 3) downsampled cell coordinates.
    loaded_zarr : da.Array
        Image data source.
    reconstruction_model : object
        Pretrained :class:`Lightsheet3DMAE` in eval mode.
    device : torch.device
        Compute device.
    head_model : torch.nn.Module
        Classification head that maps (N, C) float32 feature vectors to
        (N, n_classes) logits.  Must already be on device and in eval
        mode.
    cfg : dict
        Pipeline configuration (see :func:`_default_config`).  Uses
        head_batch_size (default 1024) to control how many cell
        features are batched before each head forward pass.
    val_transform : Callable, optional
        Optional transformation function applied to each batch chunk before encoding.
        Should take and return a numpy array of the same shape.

    Returns
    -------
    all_points : np.ndarray
        (M, 3) uint32 world coordinates of classified cells.
    all_labels : np.ndarray
        (M,) uint8 predicted class indices parallel to all_points.
    """
    head_batch_size: int = cfg.get("head_batch_size", 1024)

    all_points: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    acc_feats: list[torch.Tensor] = []
    acc_points: list[np.ndarray] = []

    def _flush() -> None:
        """Flushing the accumulated features."""
        # TODO replace this with the actual model
        batch = torch.cat(acc_feats, dim=0)
        pts = np.concatenate(acc_points, axis=0)
        with torch.no_grad():
            logits = head_model(batch)
        labels = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)
        all_points.append(pts)
        all_labels.append(labels)
        acc_feats.clear()
        acc_points.clear()

    for feats, global_points in _iter_cell_features(
        boxes, box_cells_ids, cell_zyx_ds, loaded_zarr,
        reconstruction_model, device, cfg, desc="Head inference",
        val_transform=val_transform,
    ):
        acc_feats.append(feats)
        acc_points.append(global_points)
        if sum(f.shape[0] for f in acc_feats) >= head_batch_size:
            _flush()

    if acc_feats:
        _flush()

    if not all_points:
        return np.empty((0, 3), dtype=np.uint32), np.empty(0, dtype=np.uint8)

    return np.concatenate(all_points, axis=0), np.concatenate(all_labels, axis=0)

def main(args: argparse.Namespace | None = None) -> None:
    """Run the full cell-classification pipeline.

    Parameters
    ----------
    args : argparse.Namespace, optional
        Parsed CLI arguments.  When None, arguments are read from
        sys.argv via :func:`_parse_args`.
    """
    if args is None:
        args = _parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = _default_config()

    cell_zyx_ds, loaded_zarr = load_data(
        args.proposals_path, args.dataset_path, cfg["scale"]
    )
    shape = loaded_zarr.shape

    boxes, box_cells_ids = greedy_cover_gpu(
        cell_zyx_ds, shape, box_size=cfg["box_size"], vox=cfg["vox"], verbose=True
    )

    if not boxes:
        warnings.warn("greedy_cover_gpu returned no boxes — check input coordinates.")
        return

    shuffled = np.random.permutation(len(boxes))
    boxes = [boxes[i] for i in shuffled]
    box_cells_ids = [box_cells_ids[i] for i in shuffled]

    # Validation transform to map the volumes to the
    # same distribution as the encoder was trained on
    # Note: Camilo Laiton
    # If the model was trained with percentile norm/zscore,
    # the same transform should be applied here.
    val_transform = zscore_val_augmentations()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reconstruction_model = (
        Lightsheet3DMAE.load_from_checkpoint(args.checkpoint_path, weights_only=False)
        .half()
        .to(device)
        .eval()
    )

    ipca, kmeans = train(
        boxes, box_cells_ids, cell_zyx_ds, loaded_zarr,
        reconstruction_model, device, cfg, val_transform=val_transform
    )

    if kmeans is None:
        warnings.warn(
            "k-means was never initialised (insufficient IPCA warmup data). "
            "Skipping inference.",
            RuntimeWarning,
            stacklevel=2,
        )
        return

    all_points, all_labels = infer_clustering(
        boxes, box_cells_ids, cell_zyx_ds, loaded_zarr,
        reconstruction_model, device, ipca, kmeans, cfg,
        val_transform=val_transform,
    )

    assert all_points.shape[0] == all_labels.shape[0], "Points and labels misaligned!"
    logger.info("Final dataset shape: %s %s", all_points.shape, all_labels.shape)
    if all_labels.size > 0:
        logger.info("Unique labels: %s", np.unique(all_labels))
        logger.info("Count per label: %s", np.bincount(all_labels))

    torch.save(kmeans.state_dict(), "kmeans_state_last.pt")
    logger.info("Best representative points per cluster: %s", kmeans.best_points)

    np.savez_compressed(
        "clustering_results.npz",
        labels=all_labels.astype(np.uint8),
        points=all_points.astype(np.uint32),
    )


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the pipeline.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with proposals_path, dataset_path,
        and checkpoint_path.
    """
    parser = argparse.ArgumentParser(description="Mesoscale cell classification pipeline")
    parser.add_argument("--proposals-path", required=True, help="S3 URI of the proposals CSV")
    parser.add_argument("--dataset-path", required=True, help="S3 URI of the OME-Zarr dataset")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the model checkpoint")
    return parser.parse_args()


if __name__ == "__main__":
    main()
