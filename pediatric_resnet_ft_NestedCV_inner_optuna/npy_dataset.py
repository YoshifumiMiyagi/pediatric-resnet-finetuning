"""Dataset helpers for fast memory-mapped NPY CXR input."""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NpyImageTableDataset(Dataset):
    """Read images from one .npy array using an index column in the metadata.

    Expected array shapes are (N,H,W), (N,H,W,C), or (N,C,H,W), with uint8
    preferred. The array is opened lazily with mmap_mode='r' in each process,
    which avoids repeatedly decoding PNG/JPEG files and is DataLoader-worker safe.
    """
    def __init__(self, df, npy_path, array_idx_col="array_idx", label_col="label", transform=None):
        self.df = df.reset_index(drop=True)
        self.npy_path = str(Path(npy_path).expanduser())
        self.array_idx_col = array_idx_col
        self.label_col = label_col
        self.transform = transform
        self._array = None

    def __len__(self):
        return len(self.df)

    def _get_array(self):
        if self._array is None:
            self._array = np.load(self.npy_path, mmap_mode="r")
        return self._array

    @staticmethod
    def _to_pil(x):
        x = np.asarray(x)
        if x.ndim == 2:
            return Image.fromarray(x.astype(np.uint8, copy=False), mode="L").convert("RGB")
        if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
            x = np.moveaxis(x, 0, -1)
        if x.ndim == 3 and x.shape[-1] == 1:
            x = x[..., 0]
            return Image.fromarray(x.astype(np.uint8, copy=False), mode="L").convert("RGB")
        if x.ndim == 3 and x.shape[-1] == 3:
            return Image.fromarray(x.astype(np.uint8, copy=False), mode="RGB")
        raise ValueError(f"Unsupported NPY image shape: {x.shape}")

    def __getitem__(self, i):
        row = self.df.iloc[i]
        idx = int(row[self.array_idx_col])
        x = self._to_pil(self._get_array()[idx])
        if self.transform:
            x = self.transform(x)
        y = torch.tensor(int(row[self.label_col]), dtype=torch.long)
        return x, y


def validate_npy_metadata(df, npy_path, array_idx_col="array_idx"):
    """Fail early if metadata indices cannot address the NPY array."""
    path = Path(npy_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"NPY file not found: {path}")
    if array_idx_col not in df.columns:
        raise ValueError(f"Missing CSV column: {array_idx_col}")
    if df[array_idx_col].isna().any():
        raise ValueError(f"{array_idx_col} contains missing values")
    arr = np.load(path, mmap_mode="r")
    idx = df[array_idx_col].astype(int).to_numpy()
    if len(arr) == 0 or idx.min() < 0 or idx.max() >= len(arr):
        raise ValueError(
            f"Invalid {array_idx_col}: range={idx.min()}..{idx.max()}, NPY length={len(arr)}"
        )
    return tuple(arr.shape), str(arr.dtype)
