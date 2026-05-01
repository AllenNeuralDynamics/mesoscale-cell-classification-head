"""Get index of benchmark dataset on s3."""
import os
import re

import s3fs

FNAME_RE = re.compile(r"z(\d+)y(\d+)x(\d+)Ch(\d+)")
LABEL_DIRS = ("cells", "non_cells")


def parse_s3_dataset(bucket: str) -> dict:
    """Walk an S3 bucket with structure {6-digit ID}/{3-digit channel}/{cells|non_cells}/*.tif
    and return a nested index keyed by dataset ID then channel folder.

    Returns
    -------
    dict{
            "<ID>": {
                "<channel>": {
                    "cells":     [(z, y, x), ...],
                    "non_cells": [(z, y, x), ...],
                }
            }
        }
    """
    fs = s3fs.S3FileSystem(anon=True)
    result = {}

    for id_path in fs.ls(bucket, detail=False):
        id_str = id_path.split("/")[-1]
        if not re.fullmatch(r"\d{6}", id_str):
            continue
        result[id_str] = {}

        dataset_id_path = f"{bucket}{id_str}"
        for ch_path in fs.ls(dataset_id_path, detail=False):
            ch_folder = ch_path.split("/")[-1]
            if not re.fullmatch(r"\d{3}", ch_folder):
                continue
            result[id_str][ch_folder] = {
                "cells": set(),
                "non_cells": set(),
            }

            for label in LABEL_DIRS:
                # print(f"{dataset_id_path}/{ch_folder}/{label}/*.tif")
                for fpath in fs.glob(f"{dataset_id_path}/{ch_folder}/{label}/*.tif"):
                    m = FNAME_RE.search(os.path.basename(fpath))
                    if not m:
                        continue
                    z, y, x = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    result[id_str][ch_folder][label].add((z, y, x))

            result[id_str][ch_folder]["cells"] = list(
                result[id_str][ch_folder]["cells"]
            )
            result[id_str][ch_folder]["non_cells"] = list(
                result[id_str][ch_folder]["non_cells"]
            )

    return result
