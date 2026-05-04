# mesoscale-cell-classification-head

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

Unsupervised cell classification pipeline for mesoscale 3-D light-sheet microscopy (SmartSPIM). A pretrained [Lightsheet3DMAE](https://github.com/AllenNeuralDynamics/aind-lightsheet-mae) encoder extracts patch-level features from 3-D image sub-volumes; downstream incremental PCA and online GPU k-means cluster every proposed cell without labels.

## How it works

The pipeline runs in three phases:

1. **Data loading** — cell proposals are read from an S3 CSV (`z, y, x` columns); the corresponding OME-Zarr volume is opened as a Dask array at a chosen resolution scale.
2. **Training** — a greedy density-first box cover partitions the point cloud into non-overlapping/overlapped 3-D windows. Each window is encoded by the frozen MAE; `IncrementalPCA` and `OnlineKMeans` are updated batch-by-batch.
3. **Inference** — frozen PCA + k-means assign a cluster label to every proposed cell; results are saved as `clustering_results.npz`.

## Public API

| Symbol | Module | Description |
|---|---|---|
| `OnlineKMeans` | `clustering` | GPU k-means with EMA centroid updates and dead-centroid reseeding |
| `greedy_cover_gpu` | `spatial` | Greedy density-first box cover of a 3-D point cloud (CuPy) |
| `run_batch` | `feature_extraction` | MAE encoder forward pass on a list of image chunks |
| `extract_feature_vectors_torch` | `feature_extraction` | Single-token per-cell feature lookup from a spatial feature map |
| `extract_feature_vectors_torch_3d` | `feature_extraction` | Patch-cube per-cell feature extraction (for CNN heads) |
| `pad_to_shape` | `feature_extraction` | Zero-pad a 3-D array to a target shape |

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone <repo-url>
cd mesoscale-cell-classification-head
uv sync
```

The `aind-lightsheet-mae` dependency is a private GitHub package pulled via SSH:

```bash
uv sync  # requires SSH access to AllenNeuralDynamics/aind-lightsheet-mae
```

A CUDA-capable GPU is required at runtime (`cupy-cuda12x`, `torch` with CUDA).

## CLI usage

```bash
uv run python -m mesoscale_cell_classification_head.pipeline \
    --proposals-path  s3://bucket/proposals.csv \
    --dataset-path    s3://bucket/dataset.zarr \
    --checkpoint-path /path/to/lightsheet_mae.ckpt
```

### Outputs

| File | Contents |
|---|---|
| `clustering_results.npz` | `points` (M, 3) uint32 world coordinates; `labels` (M,) uint8 cluster IDs |
| `kmeans_state_trained.pt` | Final k-means state dict (centroids, counts, best representative points) |
| `kmeans_state_<N>.pt` | Periodic checkpoints every `store_kmeans_every_n` boxes (default 500) |

### Key configuration defaults

| Parameter | Default | Description |
|---|---|---|
| `box_dim` | 128 | Encoder input side length (voxels) |
| `overlap` | 10 | Halo added to each box face for border cells |
| `n_clusters` | 4 | Number of k-means clusters |
| `n_pca_components` | 64 | PCA output dimensionality |
| `ipca_warmup_boxes` | 200 | Boxes processed before k-means is initialised |
| `max_boxes` | 30 000 | Cap on number of boxes in the training pass |
| `lr_kmeans` | 0.03 | EMA learning rate for centroid updates |

## Development

### Environment

```bash
uv sync --extra dev
```

### Code quality

```bash
# Unit tests with coverage
uv run pytest tests

# Lint
uv run ruff check

# Type check
uv run mypy src/mesoscale_cell_classification_head
```

### Documentation

```bash
uv run mkdocs serve          # live preview
uv run mkdocs build          # static site to site/
```
