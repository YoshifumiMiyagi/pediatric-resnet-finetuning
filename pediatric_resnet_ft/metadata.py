from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Union

import pandas as pd


DEFAULT_GENDER_MAP = {"F": 0, "M": 1}
DEFAULT_EXTENSIONS = (".png", ".jpg", ".jpeg")


def build_dataset_csv(
    metadata_path: Union[str, Path],
    image_dir: Union[str, Path],
    output_csv: Optional[Union[str, Path]] = None,
    sheet_name: str = "Merged",
    image_index_col: str = "Image Index",
    patient_id_col: str = "Patient ID",
    gender_col: str = "Patient Gender",
    age_col: Optional[str] = "Patient Age Clean",
    age_group_col: Optional[str] = "Age Group",
    gender_map: Optional[Mapping[str, int]] = None,
    extensions: Sequence[str] = DEFAULT_EXTENSIONS,
    recursive: bool = False,
    strict: bool = True,
) -> pd.DataFrame:
    """Create a model-ready CSV by matching files in a folder to Excel metadata.

    The image folder is treated as the source of truth for which images are used.
    No QC/final_decision filtering is applied here. This is useful when QC-selected
    images have already been copied into age-specific folders such as age_01_05.

    Output columns include patient_id, image_name, path, label and, when available,
    age and age_group. Female/Male labels default to F=0, M=1.

    Parameters
    ----------
    metadata_path:
        Excel (.xlsx/.xls) or CSV metadata table.
    image_dir:
        Directory containing the images selected for analysis.
    output_csv:
        Optional output path. If omitted, the DataFrame is returned only.
    strict:
        If True, raise an error when folder images are missing from metadata,
        metadata has duplicate Image Index values, or gender labels are invalid.
    """
    metadata_path = Path(metadata_path)
    image_dir = Path(image_dir)
    gender_map = dict(gender_map or DEFAULT_GENDER_MAP)

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    suffix = metadata_path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        metadata = pd.read_excel(metadata_path, sheet_name=sheet_name)
    elif suffix == ".csv":
        metadata = pd.read_csv(metadata_path)
    else:
        raise ValueError("metadata_path must be .xlsx, .xls, or .csv")

    required = [image_index_col, patient_id_col, gender_col]
    missing_cols = [c for c in required if c not in metadata.columns]
    if missing_cols:
        raise ValueError(f"Metadata is missing required columns: {missing_cols}")

    metadata = metadata.copy()
    metadata[image_index_col] = metadata[image_index_col].astype(str).str.strip()

    duplicate_mask = metadata[image_index_col].duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = sorted(metadata.loc[duplicate_mask, image_index_col].unique().tolist())
        message = f"Duplicate {image_index_col} values in metadata: {duplicates[:10]}"
        if strict:
            raise ValueError(message)
        metadata = metadata.drop_duplicates(subset=[image_index_col], keep="first")

    extensions = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in extensions}
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    image_paths = sorted(
        p for p in iterator if p.is_file() and p.suffix.lower() in extensions
    )
    if not image_paths:
        raise ValueError(f"No image files found in: {image_dir}")

    files = pd.DataFrame({
        "image_name": [p.name for p in image_paths],
        "path": [str(p.resolve()) for p in image_paths],
    })

    merged = files.merge(
        metadata,
        how="left",
        left_on="image_name",
        right_on=image_index_col,
        validate="one_to_one",
        indicator=True,
    )

    unmatched = merged.loc[merged["_merge"] != "both", "image_name"].tolist()
    if unmatched and strict:
        raise ValueError(
            f"{len(unmatched)} image(s) in the folder were not found in metadata. "
            f"Examples: {unmatched[:10]}"
        )
    if unmatched:
        merged = merged.loc[merged["_merge"] == "both"].copy()

    merged[gender_col] = merged[gender_col].astype(str).str.strip().str.upper()
    invalid_gender = ~merged[gender_col].isin(gender_map.keys())
    if invalid_gender.any():
        invalid_values = sorted(merged.loc[invalid_gender, gender_col].unique().tolist())
        if strict:
            raise ValueError(
                f"Unknown gender value(s): {invalid_values}. "
                f"Expected one of {list(gender_map.keys())}."
            )
        merged = merged.loc[~invalid_gender].copy()

    merged["label"] = merged[gender_col].map(gender_map).astype(int)

    # Keep patient IDs stable as strings for grouping; numeric IDs become e.g. '6147'.
    def _clean_patient_id(value):
        if pd.isna(value):
            return None
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip()

    merged["patient_id"] = merged[patient_id_col].map(_clean_patient_id)

    output_columns = ["patient_id", "image_name", "path", "label"]
    rename_map = {}
    if age_col and age_col in merged.columns:
        rename_map[age_col] = "age"
        output_columns.append("age")
    if age_group_col and age_group_col in merged.columns:
        rename_map[age_group_col] = "age_group"
        output_columns.append("age_group")

    merged = merged.rename(columns=rename_map)
    result = merged[output_columns].reset_index(drop=True)

    # A patient should not have contradictory sex labels across multiple images.
    sex_counts = result.groupby("patient_id", dropna=False)["label"].nunique()
    inconsistent = sex_counts[sex_counts > 1]
    if len(inconsistent):
        raise ValueError(
            "Inconsistent gender labels found for patient_id(s): "
            + ", ".join(map(str, inconsistent.index[:10]))
        )

    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(output_csv, index=False)

    return result


def build_age_group_csvs(
    metadata_path: Union[str, Path],
    age_group_dirs: Mapping[str, Union[str, Path]],
    output_dir: Union[str, Path] = "dataset_csvs",
    sheet_name: str = "Merged",
    **kwargs,
) -> Dict[str, pd.DataFrame]:
    """Build one CSV per age-group folder.

    Example
    -------
    build_age_group_csvs(
        metadata_path="merged_reviewer_results.xlsx",
        age_group_dirs={
            "age_01_05": "/content/drive/.../age_01_05",
            "age_06_11": "/content/drive/.../age_06_11",
            "age_12_17": "/content/drive/.../age_12_17",
        },
    )
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, pd.DataFrame] = {}
    for group_name, folder in age_group_dirs.items():
        csv_path = output_dir / f"{group_name}.csv"
        df = build_dataset_csv(
            metadata_path=metadata_path,
            image_dir=folder,
            output_csv=csv_path,
            sheet_name=sheet_name,
            **kwargs,
        )
        outputs[group_name] = df

    return outputs
