"""Module for the preprocessing steps."""

import os
import re
from typing import Dict, Hashable, List, Sequence, Tuple, Union
from urllib.parse import urlparse

import numpy as np
import torch
from monai.transforms import Compose, MapTransform, NormalizeIntensityd, ToTensord, Transform


def load_sample_percentiles(
    percentiles_dir: str,
) -> Dict[str, List[List[float]]]:
    """Load pre-computed percentiles for all samples in a directory.

    Reads every ``*_percentiles.npz`` file produced by ``compute_percentiles()``.
    The sample_id is derived by stripping ``_percentiles.npz`` from the filename.

    Parameters
    ----------
    percentiles_dir : str
        Directory containing ``*_percentiles.npz`` files.

    Returns
    -------
    Dict[str, List[List[float]]]
        ``{sample_id: [[p_low_ch0, p_high_ch0], [p_low_ch1, p_high_ch1], ...]}``
        Ready to pass directly to ``PercentileNormalizationd``.
    """
    result = {}
    for fname in os.listdir(percentiles_dir):
        if not fname.endswith("_percentiles.npz"):
            continue
        sample_id = fname[: -len("_percentiles.npz")]
        data = np.load(os.path.join(percentiles_dir, fname), allow_pickle=True)
        # combined_percentiles shape: (n_channels, 2)
        combined = data["combined_percentiles"]  # np.ndarray (n_channels, 2)
        percentiles = combined.tolist()  # [[p_low, p_high], ...]
        result[sample_id] = percentiles
        # Tile S3 paths use the raw acquisition folder name
        # (e.g. HCR_802702-s1-ls2_2026-03-13_18-00-00) while the fused volume
        # used for percentile computation has a _processed_YYYY-MM-DD suffix.
        # Register both so extract_sample_id works for tile and fused paths.
        base_id = re.sub(r"_processed_\d{4}-\d{2}-\d{2}.*$", "", sample_id)
        if base_id != sample_id:
            result[base_id] = percentiles
    return result


def extract_sample_id(s3_path: str) -> str:
    """Extract the dataset name (sample_id) from an S3 URI.

    Assumes the format ``s3://bucket/DATASET_NAME/rest/of/path``.

    Parameters
    ----------
    s3_path : str
        S3 URI, e.g.
        ``s3://aind-open-data/HCR_802702-s1-ls2_.../tiles/Tile_X_0000.zarr``

    Returns
    -------
    str
        ``HCR_802702-s1-ls2_...`` (the first path component after the bucket).
    """
    return urlparse(s3_path).path.lstrip("/").split("/")[0]


class InjectSampleIdd(Transform):
    """MONAI-style dict transform that injects a fixed ``sample_id`` into the data dict.

    Create one instance per ``ZarrDataset`` / ``MaskedZarrDataset``
    so that ``PercentileNormalizationd`` downstream can look up the
    correct percentile bounds for each sample.

    Parameters
    ----------
    sample_id : str
        Identifier that matches a key in the ``sample_percentiles`` dict
        passed to ``PercentileNormalizationd``.
    sample_id_key : str
        Key under which the id is stored. Default: ``"sample_id"``.
    """

    def __init__(
        self, sample_id: str, sample_id_key: str = "sample_id"
    ) -> None:
        """Init the transform."""
        super().__init__()
        self.sample_id = sample_id
        self.sample_id_key = sample_id_key

    def __call__(self, data: Dict) -> Dict:
        """Inject the sample_id into the data dict."""
        d = dict(data)
        d[self.sample_id_key] = self.sample_id
        return d


class PercentileNormalizationd(MapTransform):
    """MONAI-style dict transform that normalises image intensities using pre-computed percentiles.

    Looks up per-channel ``(p_low, p_high)`` bounds from ``sample_percentiles`` keyed by the
    ``sample_id`` injected upstream (e.g. by :class:`InjectSampleIdd`), clips and rescales each
    channel to ``[0, 1]``.
    """

    def __init__(
        self,
        keys: Sequence[str],
        sample_percentiles: Dict[
            Hashable, Union[List[List[float]], List[float]]
        ],
        sample_id_key: str = "sample_id",
        allow_missing_keys: bool = False,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self.sample_percentiles = sample_percentiles
        self.sample_id_key = sample_id_key

    @staticmethod
    def _parse_channel_percentiles(
        raw: Union[List[List[float]], List[float]],
    ) -> List[Tuple[float, float]]:
        """Normalise the two accepted input formats to [(p_low, p_high), ...]."""
        if not raw:
            raise ValueError("Percentile list must not be empty.")
        # Bare pair [p_low, p_high] -> wrap in a list so it looks like one channel
        if isinstance(raw[0], (int, float)):
            return [(float(raw[0]), float(raw[1]))]
        return [(float(pair[0]), float(pair[1])) for pair in raw]

    @staticmethod
    def _normalize_channel(
        channel: Union[torch.Tensor, np.ndarray],
        p_low: float,
        p_high: float,
    ) -> Union[torch.Tensor, np.ndarray]:
        """Clip to [p_low, p_high] and rescale to [0, 1]."""
        if isinstance(channel, torch.Tensor):
            out = channel.float().clamp(p_low, p_high)
            if p_high > p_low:
                out = (out - p_low) / (p_high - p_low)

            else:
                raise ValueError(
                    f"p_high must be greater than p_low, got p_low={p_low}, p_high={p_high}"
                )
        else:
            out = np.clip(channel, p_low, p_high).astype(np.float32)
            if p_high > p_low:
                out = (out - p_low) / (p_high - p_low)
            else:
                raise ValueError(
                    f"p_high must be greater than p_low, got p_low={p_low}, p_high={p_high}"
                )

        return out

    def __call__(self, data: Dict) -> Dict:
        """Apply percentile-based normalization to the specified keys."""
        d = dict(data)

        sample_id = d.get(self.sample_id_key)
        if sample_id is None:
            raise KeyError(
                f"Key '{self.sample_id_key}' not found in data dict. "
                "Pass the sample identifier alongside the image so the "
                "correct percentiles can be looked up."
            )

        if sample_id not in self.sample_percentiles:
            raise KeyError(
                f"sample_id '{sample_id}' not found in sample_percentiles. "
                f"Available ids: {list(self.sample_percentiles.keys())}"
            )

        channel_percentiles = self._parse_channel_percentiles(
            self.sample_percentiles[sample_id]
        )

        for key in self.key_iterator(d):
            img = d[key]
            has_channel_dim = img.ndim >= 4  # (C, D, H, W) or (C, H, W)

            if not has_channel_dim:
                # Treat as single channel
                p_low, p_high = channel_percentiles[0]
                d[key] = self._normalize_channel(img, p_low, p_high)
            else:
                n_channels = img.shape[0]
                if n_channels > len(channel_percentiles):
                    raise ValueError(
                        f"Image has {n_channels} channels but only "
                        f"{len(channel_percentiles)} percentile pairs were "
                        f"provided for sample '{sample_id}'."
                    )
                normalized = [
                    self._normalize_channel(img[c], *channel_percentiles[c])
                    for c in range(n_channels)
                ]
                if isinstance(img, torch.Tensor):
                    d[key] = torch.stack(normalized, dim=0)
                else:
                    d[key] = np.stack(normalized, axis=0)

        return d


def zscore_val_augmentations():
    """Version using MONAI's built-in normalization instead of custom."""
    return Compose(
        [
            NormalizeIntensityd(keys=["image"]),  # Z-score normalization
            ToTensord(keys=["image"], dtype=torch.float32),  # enforce float32
            # EnsureTyped(keys=["image"], data_type="tensor"),
        ]
    )

def compose_percentile_normalization_per_dataset(
    sample_id: str,
    sample_percentiles: Dict[str, List[List[float]]],
) -> Compose:
    """Compose a MONAI transform pipeline for percentile normalisation of one sample.

    Parameters
    ----------
    sample_id : str
        Identifier for the sample, used to look up percentile bounds.
    sample_percentiles : Dict[str, List[List[float]]]
        Dictionary mapping sample_id to percentile bounds per dataset.

    Returns
    -------
    Compose
        MONAI ``Compose`` that injects the sample_id and applies
        per-channel percentile normalisation followed by tensor conversion.
    """
    return Compose(
        [
            InjectSampleIdd(sample_id),
            PercentileNormalizationd(
                keys=["image"],
                sample_percentiles=sample_percentiles,
            ),
            ToTensord(keys=["image"], dtype=torch.float32),
        ]
    )

def apply_transform(
    batch: Sequence[np.ndarray],
    transform: Transform
):
    """Apply a MONAI transform to a batch of images.

    Parameters
    ----------
    transform : Transform
        MONAI-style transform to apply to each image in the batch.
    batch : Sequence[np.ndarray]
        List or array of images to transform.

    Returns
    -------
    torch.Tensor
        Batch of transformed images stacked into a single tensor.
    """
    return torch.stack(
        [transform({"image": img})["image"] for img in batch], dim=0
    )
