"""NPY-enabled wrapper for the extended nested-CV Optuna runner.

This keeps the tested nested-CV/Optuna logic unchanged while replacing only the
image backend with memory-mapped NPY access.
"""
import pandas as pd

from . import nested_cv_optuna_extended as _extended
from .npy_dataset import NpyImageTableDataset, validate_npy_metadata


class _ConfiguredNpyDataset(NpyImageTableDataset):
    npy_path = None
    array_idx_col = "array_idx"

    def __init__(self, df, image_col="array_idx", label_col="label", transform=None):
        # image_col is intentionally ignored: the extended runner passes its
        # image_col argument positionally, while NPY addressing uses array_idx_col.
        super().__init__(
            df=df,
            npy_path=self.npy_path,
            array_idx_col=self.array_idx_col,
            label_col=label_col,
            transform=transform,
        )


def run_nested_cv(csv_path, npy_path, array_idx_col="array_idx", **kwargs):
    """Run leakage-safe nested CV directly from a memory-mapped NPY array.

    Parameters
    ----------
    csv_path : str/path
        Metadata CSV containing label/group columns and ``array_idx_col``.
    npy_path : str/path
        NPY array. Typical pediatric CXR shape: (N, 224, 224), uint8.
    array_idx_col : str
        CSV column mapping each row to the first NPY dimension.
    **kwargs
        All arguments accepted by ``nested_cv_optuna_extended.run_nested_cv``.

    Notes
    -----
    The outer test folds remain untouched by Optuna. The NPY backend changes
    image I/O only; transforms, splitting, model construction, optimization,
    and result files are inherited from the extended runner.
    """
    df = pd.read_csv(csv_path)
    shape, dtype = validate_npy_metadata(df, npy_path, array_idx_col)

    _ConfiguredNpyDataset.npy_path = str(npy_path)
    _ConfiguredNpyDataset.array_idx_col = array_idx_col

    original_dataset = _extended.ImageTableDataset
    _extended.ImageTableDataset = _ConfiguredNpyDataset
    try:
        if kwargs.get("verbose", True):
            print(f"NPY backend: {npy_path}")
            print(f"NPY shape={shape}, dtype={dtype}, index_col={array_idx_col}")
        # The extended runner validates image_col as a CSV column and also
        # writes it to OOF output, so point image_col at array_idx_col.
        kwargs["image_col"] = array_idx_col
        return _extended.run_nested_cv(csv_path=csv_path, **kwargs)
    finally:
        _extended.ImageTableDataset = original_dataset
