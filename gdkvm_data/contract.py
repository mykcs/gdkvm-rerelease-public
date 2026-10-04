"""Scientific data contract helpers.

These functions own split and temporal-sampling semantics without depending on
a particular storage backend.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


CAMUS_EXPECTED_PATIENTS = {"train": 400, "val": 50, "test": 50}
ECHONET_EXPECTED_VIDEOS = {"TRAIN": 7465, "VAL": 1288, "TEST": 1277}
ECHONET_EXPECTED_ANNOTATED_VIDEOS = {"TRAIN": 7460, "VAL": 1288, "TEST": 1276}


def uniform_frame_indices(
    total_frames: int,
    count: int = 10,
    *,
    start: int = 0,
    stop: int | None = None,
) -> np.ndarray:
    """Return deterministic, uniformly spaced frame indices including endpoints.

    The rerelease defines uniform sampling as linear spacing over an inclusive
    interval followed by numpy round-to-nearest. The interval must contain at
    least count distinct integer frames.
    """
    total_frames = int(total_frames)
    count = int(count)
    start = int(start)
    stop = total_frames - 1 if stop is None else int(stop)

    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    if count < 2:
        raise ValueError("count must be >= 2")
    if start < 0 or stop >= total_frames or start > stop:
        raise ValueError("invalid inclusive sampling interval")
    if stop - start + 1 < count:
        raise ValueError("sampling interval has fewer distinct frames than requested")

    indices = np.rint(np.linspace(start, stop, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise ValueError("uniform sampling produced duplicate indices")
    if indices[0] != start or indices[-1] != stop:
        raise AssertionError("uniform sampling must preserve endpoints")
    return indices


def uniform_cyclic_frame_indices(
    total_frames: int,
    start: int,
    stop: int,
    count: int = 10,
) -> np.ndarray:
    """Uniformly sample the forward cyclic interval from start through stop.

    This makes the cardiac-cycle convention explicit when ED occurs after ES in
    absolute video indexing: ED is first, ES is last, and sampling proceeds
    forward through the end of the video and wraps to frame zero if required.
    """
    total_frames = int(total_frames)
    start = int(start)
    stop = int(stop)
    count = int(count)
    if total_frames < 1:
        raise ValueError("total_frames must be >= 1")
    if not (0 <= start < total_frames and 0 <= stop < total_frames):
        raise ValueError("start/stop must be valid frame indices")
    forward_distance = (stop - start) % total_frames
    if forward_distance < 1:
        raise ValueError("ED and ES must be distinct frames")

    # Sample continuous positions uniformly, then map them to the nearest
    # discrete video frame. A short ED→ES interval may therefore repeat a
    # discrete frame. This preserves the fixed 10-frame model contract without
    # discarding valid studies whose systolic interval has fewer than 10
    # distinct frames.
    offsets = np.rint(np.linspace(0, forward_distance, count)).astype(np.int64)
    indices = (start + offsets) % total_frames
    if indices[0] != start or indices[-1] != stop:
        raise AssertionError("cyclic sampling must preserve ED/ES endpoints")
    if np.any((indices < 0) | (indices >= total_frames)):
        raise AssertionError("cyclic sampling produced an invalid frame index")
    return indices.astype(np.int64)


def validate_disjoint_splits(
    splits: Mapping[str, Sequence[str]],
    *,
    expected_counts: Mapping[str, int] | None = None,
) -> dict[str, int]:
    normalized = {name: [str(x).strip() for x in values] for name, values in splits.items()}
    seen: dict[str, str] = {}
    counts: dict[str, int] = {}

    for name, values in normalized.items():
        if any(not value for value in values):
            raise ValueError(f"{name} contains an empty ID")
        if len(values) != len(set(values)):
            raise ValueError(f"{name} contains duplicate IDs")
        counts[name] = len(values)
        for value in values:
            if value in seen:
                raise ValueError(f"ID {value} appears in both {seen[value]} and {name}")
            seen[value] = name

    if expected_counts is not None:
        for name, expected in expected_counts.items():
            actual = counts.get(name)
            if actual != expected:
                raise ValueError(f"{name} count {actual} != expected {expected}")
    return counts


def _read_nonempty_lines(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_camus_split_dir(root: str | Path) -> dict[str, list[str]]:
    root = Path(root)
    mapping = {
        "train": _read_nonempty_lines(root / "subgroup_training.txt"),
        "val": _read_nonempty_lines(root / "subgroup_validation.txt"),
        "test": _read_nonempty_lines(root / "subgroup_testing.txt"),
    }
    validate_disjoint_splits(mapping, expected_counts=CAMUS_EXPECTED_PATIENTS)
    return mapping


def load_echonet_filelist(path: str | Path) -> dict[str, list[str]]:
    groups = {"TRAIN": [], "VAL": [], "TEST": []}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"FileName", "Split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("EchoNet FileList.csv must contain FileName and Split")
        for row in reader:
            split = str(row["Split"]).strip().upper()
            filename = str(row["FileName"]).strip()
            if split not in groups:
                raise ValueError(f"unexpected EchoNet split {split!r}")
            if not filename:
                raise ValueError("empty EchoNet FileName")
            groups[split].append(filename)

    validate_disjoint_splits(groups, expected_counts=ECHONET_EXPECTED_VIDEOS)
    return groups


def expand_camus_views(patient_splits: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    return {
        split: [
            f"{patient}_{view}"
            for patient in patients
            for view in ("2CH", "4CH")
        ]
        for split, patients in patient_splits.items()
    }
