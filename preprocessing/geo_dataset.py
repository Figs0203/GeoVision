from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


# Geographical binning constants
LAT_BIN_SIZE = 10  # degrees
LON_BIN_SIZE = 10  # degrees
NUM_LAT_BINS = int(180 / LAT_BIN_SIZE)  # 18 bins to cover [-90, 90)
NUM_LON_BINS = int(360 / LON_BIN_SIZE)  # 36 bins to cover [-180, 180)
UNKNOWN_BIN = -1


def compute_lat_bin(lat: float) -> int:
    if pd.isna(lat):
        return UNKNOWN_BIN
    value = int(np.floor((lat + 90.0) / LAT_BIN_SIZE))
    return max(0, min(NUM_LAT_BINS - 1, value))


def compute_lon_bin(lon: float) -> int:
    if pd.isna(lon):
        return UNKNOWN_BIN
    value = int(np.floor((lon + 180.0) / LON_BIN_SIZE))
    return max(0, min(NUM_LON_BINS - 1, value))


def load_clip_embeddings(npz_path: Path) -> Tuple[Dict[str, np.ndarray], int, Optional[str]]:
    data = np.load(npz_path, allow_pickle=True)
    embeddings = data["embeddings"]
    paths = data["paths"]
    model_name = None
    if "model_name" in data:
        model_name = str(np.array(data["model_name"]).item())
    mapping: Dict[str, np.ndarray] = {str(path): emb for path, emb in zip(paths, embeddings)}
    return mapping, embeddings.shape[1], model_name


@dataclass
class MetadataRecord:
    relative_path: Path
    label_name: str
    scene_type: str
    continent: str
    lat_bin: int
    lon_bin: int


class GeoImageDataset(Dataset):
    """PyTorch dataset that loads images along with geographic bin indices and optional CLIP embeddings."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        root_dir: Path,
        transforms,
        label_to_id: Dict[str, int],
        lat_bins_total: int,
        lon_bins_total: int,
        clip_embeddings: Optional[Dict[str, np.ndarray]] = None,
        clip_dim: Optional[int] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.transforms = transforms
        self.label_to_id = label_to_id
        self.lat_bins_total = lat_bins_total
        self.lon_bins_total = lon_bins_total
        self.clip_embeddings = clip_embeddings
        self.clip_dim = clip_dim
        self.records: List[MetadataRecord] = []

        if self.clip_embeddings is not None and self.clip_dim is None:
            try:
                first_key = next(iter(self.clip_embeddings))
                self.clip_dim = len(self.clip_embeddings[first_key])
            except StopIteration:
                self.clip_dim = 0
        if self.clip_embeddings is None:
            self.clip_dim = 0

        for row in metadata.to_dict("records"):
            rel_path = Path(row["relative_path"])
            label_name = row["label_name"]
            lat_bin = int(row.get("lat_bin", UNKNOWN_BIN))
            lon_bin = int(row.get("lon_bin", UNKNOWN_BIN))

            self.records.append(
                MetadataRecord(
                    relative_path=rel_path,
                    label_name=label_name,
                    scene_type=row.get("scene_type", ""),
                    continent=row.get("continent", ""),
                    lat_bin=lat_bin,
                    lon_bin=lon_bin,
                )
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        record = self.records[idx]
        image_path = self.root_dir / record.relative_path
        with Image.open(image_path) as img:
            image = img.convert("RGB")

        np_image = np.array(image)
        if self.transforms is not None:
            aug = self.transforms(image=np_image)
            tensor = aug["image"]
        else:
            tensor = ToTensorV2()(image=np_image)["image"]

        label = self.label_to_id[record.label_name]
        lat_idx = record.lat_bin if record.lat_bin >= 0 else self.lat_bins_total - 1
        lon_idx = record.lon_bin if record.lon_bin >= 0 else self.lon_bins_total - 1

        sample: Dict[str, torch.Tensor] = {
            "pixel_values": tensor,
            "labels": label,
            "lat_bins": lat_idx,
            "lon_bins": lon_idx,
        }

        if self.clip_dim and self.clip_embeddings is not None:
            key = str(record.relative_path)
            emb = self.clip_embeddings.get(key)
            if emb is None:
                clip_tensor = torch.zeros(self.clip_dim, dtype=torch.float32)
            else:
                clip_tensor = torch.from_numpy(np.asarray(emb)).float()
            sample["clip_embeddings"] = clip_tensor

        return sample


