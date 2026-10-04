"""EchoNet-Dynamic conversion helpers for the GDKVM re-release."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np

from .contract import (
    ECHONET_EXPECTED_ANNOTATED_VIDEOS,
    ECHONET_EXPECTED_VIDEOS,
    load_echonet_filelist,
    uniform_cyclic_frame_indices,
)


TraceRow = tuple[float, float, float, float]
TraceMap = dict[str, dict[int, list[TraceRow]]]


def with_avi(name: str) -> str:
    value = str(name).strip()
    return value if value.lower().endswith(".avi") else f"{value}.avi"


def load_tracings(path: str | Path) -> TraceMap:
    traces: TraceMap = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = ["FileName", "X1", "Y1", "X2", "Y2", "Frame"]
        if reader.fieldnames != required:
            raise ValueError(f"unexpected VolumeTracings header: {reader.fieldnames}")
        for row in reader:
            filename = str(row["FileName"]).strip()
            frame = int(row["Frame"])
            traces.setdefault(filename, {}).setdefault(frame, []).append(
                (
                    float(row["X1"]),
                    float(row["Y1"]),
                    float(row["X2"]),
                    float(row["Y2"]),
                )
            )
    return traces


def polygon_from_trace(rows: list[TraceRow], shape: tuple[int, int]) -> np.ndarray:
    if len(rows) < 2:
        raise ValueError("trace requires at least two chord rows")
    arr = np.asarray(rows, dtype=np.float64)
    x1, y1, x2, y2 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    # Match the official EchoNet polygon construction convention: the first
    # chord is the long-axis line and the remaining chord endpoints trace the
    # two sides of the endocardial contour.
    x = np.concatenate((x1[1:], np.flip(x2[1:])))
    y = np.concatenate((y1[1:], np.flip(y2[1:])))
    points = np.stack((np.rint(x), np.rint(y)), axis=1).astype(np.int32)
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def resize_gray(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.ndim == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(
        frame, (size, size), interpolation=cv2.INTER_LINEAR
    ).astype(np.uint8)


def resize_mask(mask: np.ndarray, size: int) -> np.ndarray:
    return cv2.resize(
        mask, (size, size), interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)


def video_metadata(video_path: str | Path) -> tuple[int, tuple[int, int]]:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if total < 1 or width < 1 or height < 1:
        raise RuntimeError(f"invalid video metadata: {video_path}")
    return total, (height, width)


def read_selected_frames(video_path: str | Path, indices: np.ndarray) -> np.ndarray:
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    output: list[np.ndarray] = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"cannot read frame {int(index)}: {video_path}")
            output.append(frame)
    finally:
        cap.release()
    return np.stack(output)


def convert_echonet_video(
    video_path: str | Path,
    frame_traces: Mapping[int, list[TraceRow]],
    *,
    size: int = 128,
    frame_count: int = 10,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Convert one authorized EchoNet video to the fixed re-release clip contract."""
    video_path = Path(video_path)
    if len(frame_traces) != 2:
        raise ValueError(
            f"expected exactly two traced frames for {video_path.name}, "
            f"got {len(frame_traces)}"
        )

    total_frames, native_shape = video_metadata(video_path)
    annotated = {
        int(frame_index): polygon_from_trace(trace_rows, native_shape)
        for frame_index, trace_rows in frame_traces.items()
    }
    for frame_index in annotated:
        if not 0 <= frame_index < total_frames:
            raise ValueError(
                f"traced frame {frame_index} outside video with {total_frames} frames"
            )

    areas = {frame_index: int(mask.sum()) for frame_index, mask in annotated.items()}
    ordered = sorted(areas, key=lambda index: areas[index])
    es_index = int(ordered[0])
    ed_index = int(ordered[-1])
    if areas[ed_index] == areas[es_index]:
        raise ValueError("cannot infer ED/ES because traced LV areas are equal")

    indices = uniform_cyclic_frame_indices(
        total_frames, ed_index, es_index, frame_count
    )
    selected = read_selected_frames(video_path, indices)

    frames = np.stack([resize_gray(frame, size) for frame in selected])
    masks = np.zeros((frame_count, size, size), dtype=np.uint8)
    masks[0] = resize_mask(annotated[ed_index], size)
    masks[-1] = resize_mask(annotated[es_index], size)

    metadata: dict[str, object] = {
        "source_frame_count": total_frames,
        "native_shape": list(native_shape),
        "ed_index": ed_index,
        "es_index": es_index,
        "sampled_indices": [int(index) for index in indices],
        "repeated_sampled_frames": int(frame_count - len(np.unique(indices))),
        "temporal_contract": (
            "forward-cyclic-uniform-ed-to-es-inclusive-nearest-frame"
        ),
        "annotation_contract": "ED/ES masks only; intermediate masks zero",
    }
    return frames, masks, metadata


def build_echonet_npy_cache(
    source: str | Path,
    output: str | Path,
    *,
    split: str | None = None,
    size: int = 128,
    frame_count: int = 10,
    limit_samples: int | None = None,
) -> Path:
    """Build a contiguous uint8 NPY cache from an authorized EchoNet source."""
    source = Path(source)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if limit_samples is not None and limit_samples < 1:
        raise ValueError("limit_samples must be >= 1")
    split = split.upper() if split is not None else None
    if split is not None and split not in {"TRAIN", "VAL", "TEST"}:
        raise ValueError(f"invalid split: {split}")

    filelist = source / "FileList.csv"
    tracings_path = source / "VolumeTracings.csv"
    videos_root = source / "Videos"
    for path in (filelist, tracings_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing required EchoNet file: {path}")
    if not videos_root.is_dir():
        raise FileNotFoundError(f"missing EchoNet Videos directory: {videos_root}")

    splits = load_echonet_filelist(filelist)
    traces = load_tracings(tracings_path)
    trace_files = set(traces)
    video_files = {path.name: path for path in videos_root.glob("*.avi")}

    filelist_video_to_key = {
        with_avi(name): name for names in splits.values() for name in names
    }
    filelist_videos = set(filelist_video_to_key)
    filelist_without_traces = sorted(filelist_videos - trace_files)
    orphan_traces = sorted(trace_files - filelist_videos)
    filelist_without_video_count = len(filelist_videos - set(video_files))

    eligible_by_split: dict[str, list[str]] = {}
    for split_name, names in splits.items():
        eligible_by_split[split_name] = [
            name
            for name in names
            if with_avi(name) in trace_files and with_avi(name) in video_files
        ]

    source_is_full_video_tree = len(video_files) == sum(ECHONET_EXPECTED_VIDEOS.values())
    if source_is_full_video_tree:
        if filelist_without_video_count:
            raise ValueError(
                f"full EchoNet source is missing {filelist_without_video_count} "
                "FileList videos"
            )
        observed = {key: len(value) for key, value in eligible_by_split.items()}
        if observed != ECHONET_EXPECTED_ANNOTATED_VIDEOS:
            raise ValueError(
                "unexpected EchoNet tracing coverage: "
                f"{observed} != {ECHONET_EXPECTED_ANNOTATED_VIDEOS}"
            )

    selected_splits = (split,) if split else ("TRAIN", "VAL", "TEST")
    samples = [
        (split_name, name)
        for split_name in selected_splits
        for name in eligible_by_split[split_name]
    ]
    if limit_samples is not None:
        samples = samples[:limit_samples]
    if not samples:
        raise ValueError("no eligible EchoNet samples available")

    output.mkdir(parents=True)
    shape = (len(samples), frame_count, size, size)
    frames_mm = np.lib.format.open_memmap(
        output / "frames.npy", mode="w+", dtype=np.uint8, shape=shape
    )
    masks_mm = np.lib.format.open_memmap(
        output / "masks.npy", mode="w+", dtype=np.uint8, shape=shape
    )
    metadata_rows: list[dict[str, object]] = []

    try:
        for row_index, (split_name, file_key) in enumerate(samples):
            filename = with_avi(file_key)
            frames, masks, metadata = convert_echonet_video(
                video_files[filename],
                traces[filename],
                size=size,
                frame_count=frame_count,
            )
            frames_mm[row_index] = frames
            masks_mm[row_index] = masks
            metadata_rows.append(
                {
                    "sample_id": file_key,
                    "filename": filename,
                    "split": split_name,
                    **metadata,
                }
            )
            if (row_index + 1) % 100 == 0:
                print(f"converted {row_index + 1}/{len(samples)}", flush=True)
    finally:
        frames_mm.flush()
        masks_mm.flush()
        del frames_mm, masks_mm

    with (output / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    manifest = {
        "cache_version": 1,
        "dataset": "EchoNet-Dynamic",
        "data_protocol_id": "gdkvm-rerelease-v1-data",
        "format": "contiguous-npy",
        "shape": list(shape),
        "dtype": "uint8",
        "image_resize": "OpenCV bilinear",
        "mask_resize": "OpenCV nearest",
        "split_source": "FileList.csv",
        "trace_source": "VolumeTracings.csv",
        "temporal_sampling": (
            "forward cyclic uniform ED-to-ES inclusive; "
            "nearest discrete frame; repeats allowed"
        ),
        "annotation_policy": "ED/ES only",
        "selection": {"split": split, "limit_samples": limit_samples},
        "source_video_count": len(video_files),
        "source_is_full_video_tree": source_is_full_video_tree,
        "filelist_split_counts": {key: len(value) for key, value in splits.items()},
        "eligible_annotated_counts": {
            key: len(value) for key, value in eligible_by_split.items()
        },
        "filelist_without_traces": filelist_without_traces,
        "orphan_traces": orphan_traces,
        "filelist_without_video_count": filelist_without_video_count,
        "converted_samples": len(samples),
        "samples_with_repeated_frames": sum(
            1
            for row in metadata_rows
            if int(row["repeated_sampled_frames"]) > 0
        ),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output
