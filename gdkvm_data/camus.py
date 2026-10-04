"""CAMUS conversion helpers for official archive and canonical NIfTI layouts."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Iterable, Mapping
import zipfile

import nibabel as nib
import numpy as np
from PIL import Image

from .contract import (
    CAMUS_EXPECTED_PATIENTS,
    uniform_frame_indices,
    validate_disjoint_splits,
)
from .npy_cache import write_npy_cache


_SPLIT_FILES = {
    "subgroup_training.txt": "train",
    "subgroup_validation.txt": "val",
    "subgroup_testing.txt": "test",
}


def _validate_splits(splits: dict[str, list[str]]) -> dict[str, list[str]]:
    if set(splits) != {"train", "val", "test"}:
        raise ValueError(f"incomplete CAMUS split: {sorted(splits)}")
    validate_disjoint_splits(splits, expected_counts=CAMUS_EXPECTED_PATIENTS)
    return splits


def load_split_archive(path: str | Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            name = Path(member).name
            if name not in _SPLIT_FILES:
                continue
            splits[_SPLIT_FILES[name]] = [
                line.strip()
                for line in archive.read(member).decode("utf-8").splitlines()
                if line.strip()
            ]
    return _validate_splits(splits)


def load_split_directory(path: str | Path) -> dict[str, list[str]]:
    """Load the canonical CAMUS database_split directory."""
    path = Path(path)
    splits: dict[str, list[str]] = {}
    for filename, split_name in _SPLIT_FILES.items():
        split_path = path / filename
        if not split_path.is_file():
            continue
        splits[split_name] = [
            line.strip()
            for line in split_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return _validate_splits(splits)


def _as_uint8_frames(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if not np.isfinite(array).all():
        raise ValueError("CAMUS image contains non-finite values")
    rounded = np.rint(array)
    if rounded.min() < 0 or rounded.max() > 255:
        raise ValueError("CAMUS image is outside uint8 intensity range")
    return rounded.astype(np.uint8)


def _as_uint8_labels(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array)
    if not np.isfinite(array).all():
        raise ValueError("CAMUS mask contains non-finite values")
    rounded = np.rint(array)
    if not np.allclose(array, rounded, atol=1e-6):
        raise ValueError("CAMUS mask contains non-integer labels")
    if rounded.min() < 0 or rounded.max() > 255:
        raise ValueError("CAMUS labels are outside uint8 range")
    return rounded.astype(np.uint8)


def _resize_sequence(
    sequence_hwt: np.ndarray,
    size: int,
    *,
    is_mask: bool,
) -> np.ndarray:
    if sequence_hwt.ndim != 3:
        raise ValueError("CAMUS half sequence must be H x W x T")
    resample = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    out = np.empty((sequence_hwt.shape[2], size, size), dtype=np.uint8)
    for index in range(sequence_hwt.shape[2]):
        image = Image.fromarray(sequence_hwt[:, :, index])
        out[index] = np.asarray(
            image.resize((size, size), resample=resample),
            dtype=np.uint8,
        )
    return out


def _convert_view(
    image_path: Path,
    mask_path: Path,
    *,
    patient_id: str,
    view: str,
    size: int,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    image_nii = nib.load(str(image_path))
    mask_nii = nib.load(str(mask_path))
    image = _as_uint8_frames(np.asarray(image_nii.dataobj))
    mask = _as_uint8_labels(np.asarray(mask_nii.dataobj))
    if image.shape != mask.shape:
        raise ValueError(f"{patient_id} {view} image/mask shape mismatch")

    indices = uniform_frame_indices(image.shape[2], frame_count)
    frames = _resize_sequence(image[:, :, indices], size, is_mask=False)
    masks = _resize_sequence(mask[:, :, indices], size, is_mask=True)

    zooms = image_nii.header.get_zooms()
    metadata = {
        "sample_id": f"{patient_id}_{view}",
        "patient_id": patient_id,
        "view": view,
        "source_shape_hwt": list(image.shape),
        "source_spacing_axis01": [
            round(float(zooms[0]), 6),
            round(float(zooms[1]), 6),
        ],
        "sampled_frame_indices_zero_based": [int(i) for i in indices],
        "resize_hw": [size, size],
        "resized_spacing_axis01": [
            round(float(zooms[0]) * image.shape[0] / size, 6),
            round(float(zooms[1]) * image.shape[1] / size, 6),
        ],
        "mask_labels_present": [int(v) for v in np.unique(masks)],
    }
    return frames, masks, metadata


def convert_patient_directory(
    patient_dir: str | Path,
    *,
    patient_id: str | None = None,
    size: int = 256,
    frame_count: int = 10,
    views: Iterable[str] = ("2CH", "4CH"),
) -> list[tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    """Convert one patient from the canonical unpacked database_nifti layout."""
    patient_dir = Path(patient_dir)
    patient_id = str(patient_id or patient_dir.name)
    rows = []
    for view in views:
        image_path = patient_dir / f"{patient_id}_{view}_half_sequence.nii.gz"
        mask_path = patient_dir / f"{patient_id}_{view}_half_sequence_gt.nii.gz"
        missing = [str(path) for path in (image_path, mask_path) if not path.is_file()]
        if missing:
            raise ValueError(f"{patient_id} {view} missing files: {missing}")
        rows.append(
            _convert_view(
                image_path,
                mask_path,
                patient_id=patient_id,
                view=view,
                size=size,
                frame_count=frame_count,
            )
        )
    return rows


def convert_patient_archive(
    archive_path: str | Path,
    *,
    patient_id: str,
    size: int = 256,
    frame_count: int = 10,
    views: Iterable[str] = ("2CH", "4CH"),
) -> list[tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    """Convert one per-patient archive produced by download_camus.py."""
    archive_path = Path(archive_path)
    views = tuple(views)
    with tempfile.TemporaryDirectory(prefix=f"{patient_id}-camus-") as tmp:
        tmp_root = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            names = set(archive.namelist())
            for view in views:
                for suffix in ("half_sequence.nii.gz", "half_sequence_gt.nii.gz"):
                    rel = f"{patient_id}/{patient_id}_{view}_{suffix}"
                    if rel not in names:
                        raise ValueError(f"{archive_path.name} missing {rel}")
                    archive.extract(rel, tmp_root)
        return convert_patient_directory(
            tmp_root / patient_id,
            patient_id=patient_id,
            size=size,
            frame_count=frame_count,
            views=views,
        )


def _write_cache(
    output_root: str | Path,
    rows: list[tuple[np.ndarray, np.ndarray, dict[str, object]]],
    *,
    split: str,
    size: int,
    frame_count: int,
    patient_count: int,
    partial_source: bool,
    source_layout: str,
) -> Path:
    if not rows:
        raise ValueError("no CAMUS samples were converted")
    frames_array = np.stack([row[0] for row in rows])
    masks_array = np.stack([row[1] for row in rows])
    metadata_rows: list[Mapping[str, object]] = [row[2] for row in rows]
    return write_npy_cache(
        output_root,
        frames_array,
        masks_array,
        metadata_rows,
        manifest_extra={
            "dataset": "CAMUS",
            "split": split,
            "frame_sampling": "uniform-inclusive-endpoints-rint-v1",
            "frame_count": frame_count,
            "resize_hw": [size, size],
            "mask_semantics": "preserve-official-multiclass-labels",
            "partial_source": bool(partial_source),
            "patient_count": patient_count,
            "source_layout": source_layout,
        },
    )


def build_npy_cache(
    source_root: str | Path,
    output_root: str | Path,
    *,
    split: str,
    size: int = 256,
    frame_count: int = 10,
    patient_limit: int | None = None,
    available_only: bool = False,
) -> Path:
    """Build from the per-patient ZIP layout produced by download_camus.py."""
    source_root = Path(source_root)
    splits = load_split_archive(source_root / "database_split.zip")
    if split not in splits:
        raise ValueError(f"unknown CAMUS split {split!r}")
    patients = list(splits[split])
    if available_only:
        patients = [
            patient_id
            for patient_id in patients
            if (source_root / "patients" / f"{patient_id}.zip").is_file()
        ]
    if patient_limit is not None:
        patients = patients[:patient_limit]

    rows = []
    for patient_id in patients:
        archive = source_root / "patients" / f"{patient_id}.zip"
        if not archive.is_file():
            raise FileNotFoundError(f"missing CAMUS patient archive: {archive}")
        rows.extend(
            convert_patient_archive(
                archive,
                patient_id=patient_id,
                size=size,
                frame_count=frame_count,
            )
        )

    return _write_cache(
        output_root,
        rows,
        split=split,
        size=size,
        frame_count=frame_count,
        patient_count=len(patients),
        partial_source=bool(available_only or patient_limit is not None),
        source_layout="per-patient-zip",
    )


def build_npy_cache_from_nifti_root(
    source_root: str | Path,
    output_root: str | Path,
    *,
    split: str,
    size: int = 256,
    frame_count: int = 10,
    patient_limit: int | None = None,
    available_only: bool = False,
) -> Path:
    """Build directly from canonical CAMUS_public/database_nifti."""
    source_root = Path(source_root)
    split_dir = source_root / "database_split"
    nifti_root = source_root / "database_nifti"
    if not split_dir.is_dir() or not nifti_root.is_dir():
        raise ValueError(
            "canonical CAMUS source must contain database_split/ and database_nifti/"
        )

    splits = load_split_directory(split_dir)
    if split not in splits:
        raise ValueError(f"unknown CAMUS split {split!r}")
    patients = list(splits[split])
    if available_only:
        patients = [
            patient_id
            for patient_id in patients
            if (nifti_root / patient_id).is_dir()
        ]
    if patient_limit is not None:
        patients = patients[:patient_limit]

    rows = []
    for patient_id in patients:
        patient_dir = nifti_root / patient_id
        if not patient_dir.is_dir():
            raise FileNotFoundError(f"missing CAMUS patient directory: {patient_dir}")
        rows.extend(
            convert_patient_directory(
                patient_dir,
                patient_id=patient_id,
                size=size,
                frame_count=frame_count,
            )
        )

    return _write_cache(
        output_root,
        rows,
        split=split,
        size=size,
        frame_count=frame_count,
        patient_count=len(patients),
        partial_source=bool(available_only or patient_limit is not None),
        source_layout="canonical-nifti-directory",
    )


def build_npy_cache_auto(
    source_root: str | Path,
    output_root: str | Path,
    **kwargs,
) -> Path:
    """Auto-detect canonical NIfTI versus per-patient ZIP source layout."""
    source_root = Path(source_root)
    if (
        (source_root / "database_nifti").is_dir()
        and (source_root / "database_split").is_dir()
    ):
        return build_npy_cache_from_nifti_root(
            source_root,
            output_root,
            **kwargs,
        )
    return build_npy_cache(source_root, output_root, **kwargs)
