#!/usr/bin/env python3
"""Compare legacy image-file and modern cache layouts with explicit access patterns."""

from __future__ import annotations

import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import random
import tarfile
import time
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_bench import environment_snapshot, write_receipt


def timer(fn, repeats: int, sample_count: int) -> dict[str, float | int]:
    values: list[float] = []
    checksum = 0
    for _ in range(repeats):
        start = time.perf_counter()
        checksum ^= int(fn())
        values.append(time.perf_counter() - start)
    mean_s = float(np.mean(values))
    return {
        "n": repeats,
        "mean_s": mean_s,
        "median_s": float(np.median(values)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
        "samples_per_s": float(sample_count / max(mean_s, 1e-12)),
        "checksum": checksum,
    }


def canonical_hash(frames: np.ndarray, masks: np.ndarray) -> str:
    digest = sha256()
    for sample in range(len(frames)):
        digest.update(np.asarray(frames[sample]).tobytes(order="C"))
        digest.update(np.asarray(masks[sample]).tobytes(order="C"))
    return digest.hexdigest()


def save_legacy_jpeg(root: Path, frames: np.ndarray, masks: np.ndarray) -> None:
    root.mkdir(parents=True)
    for sample in range(len(frames)):
        sample_root = root / f"{sample:05d}"
        (sample_root / "img").mkdir(parents=True)
        (sample_root / "mask").mkdir()
        for frame in range(frames.shape[1]):
            Image.fromarray(frames[sample, frame]).save(
                sample_root / "img" / f"{frame:03d}.jpg",
                quality=95,
            )
            Image.fromarray(masks[sample, frame]).save(
                sample_root / "mask" / f"{frame:03d}.png"
            )


def read_legacy(root: Path, order: list[int], frames_per_sample: int) -> int:
    checksum = 0
    for sample in order:
        sample_root = root / f"{sample:05d}"
        for frame in range(frames_per_sample):
            checksum ^= int(
                np.asarray(
                    Image.open(sample_root / "img" / f"{frame:03d}.jpg")
                ).sum()
            )
            checksum ^= int(
                np.asarray(
                    Image.open(sample_root / "mask" / f"{frame:03d}.png")
                ).sum()
            )
    return checksum


def legacy_hash(root: Path, sample_count: int, frames_per_sample: int) -> str:
    digest = sha256()
    for sample in range(sample_count):
        sample_root = root / f"{sample:05d}"
        frames = []
        masks = []
        for frame in range(frames_per_sample):
            frames.append(
                np.asarray(
                    Image.open(sample_root / "img" / f"{frame:03d}.jpg"),
                    dtype=np.uint8,
                )
            )
            masks.append(
                np.asarray(
                    Image.open(sample_root / "mask" / f"{frame:03d}.png"),
                    dtype=np.uint8,
                )
            )
        digest.update(np.stack(frames).tobytes(order="C"))
        digest.update(np.stack(masks).tobytes(order="C"))
    return digest.hexdigest()


def read_npy(root: Path, order: list[int]) -> int:
    frames = np.load(root / "frames.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(root / "masks.npy", mmap_mode="r", allow_pickle=False)
    checksum = 0
    for sample in order:
        checksum ^= int(np.asarray(frames[sample]).sum())
        checksum ^= int(np.asarray(masks[sample]).sum())
    return checksum


def npy_hash(root: Path, sample_count: int) -> str:
    frames = np.load(root / "frames.npy", mmap_mode="r", allow_pickle=False)
    masks = np.load(root / "masks.npy", mmap_mode="r", allow_pickle=False)
    return canonical_hash(frames[:sample_count], masks[:sample_count])


def save_tar(path: Path, frames: np.ndarray, masks: np.ndarray) -> None:
    with tarfile.open(path, "w") as archive:
        for sample in range(len(frames)):
            for kind, array in (("frames", frames[sample]), ("masks", masks[sample])):
                buffer = BytesIO()
                np.save(buffer, array, allow_pickle=False)
                payload = buffer.getvalue()
                info = tarfile.TarInfo(f"{sample:05d}.{kind}.npy")
                info.size = len(payload)
                archive.addfile(info, BytesIO(payload))


def read_tar(path: Path) -> int:
    checksum = 0
    with tarfile.open(path, "r") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".npy"):
                continue
            stream = archive.extractfile(member)
            if stream is not None:
                payload = BytesIO(stream.read())
                checksum ^= int(np.load(payload, allow_pickle=False).sum())
    return checksum


def tar_hash(path: Path) -> str:
    digest = sha256()
    with tarfile.open(path, "r") as archive:
        for member in archive:
            if not member.isfile() or not member.name.endswith(".npy"):
                continue
            stream = archive.extractfile(member)
            if stream is not None:
                payload = BytesIO(stream.read())
                digest.update(np.load(payload, allow_pickle=False).tobytes(order="C"))
    return digest.hexdigest()


def save_zarr(root: Path, frames: np.ndarray, masks: np.ndarray) -> str | None:
    try:
        import zarr
    except ImportError:
        return "zarr not installed"
    group = zarr.open_group(str(root), mode="w")
    chunks = (min(16, len(frames)), 1, frames.shape[2], frames.shape[3])
    group.create_array("frames", data=frames, chunks=chunks)
    group.create_array("masks", data=masks, chunks=chunks)
    return None


def read_zarr(root: Path, order: list[int]) -> int:
    import zarr

    group = zarr.open_group(str(root), mode="r")
    checksum = 0
    for sample in order:
        checksum ^= int(np.asarray(group["frames"][sample]).sum())
        checksum ^= int(np.asarray(group["masks"][sample]).sum())
    return checksum


def zarr_hash(root: Path, sample_count: int) -> str:
    import zarr

    group = zarr.open_group(str(root), mode="r")
    digest = sha256()
    for sample in range(sample_count):
        digest.update(np.asarray(group["frames"][sample]).tobytes(order="C"))
        digest.update(np.asarray(group["masks"][sample]).tobytes(order="C"))
    return digest.hexdigest()


def tree_bytes(root: Path) -> int:
    if root.is_file():
        return root.stat().st_size
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--dataset-id", default="synthetic-or-camus-cache")
    parser.add_argument("--split-id", default="fixture")
    args = parser.parse_args()

    if args.workspace.exists():
        raise SystemExit(f"workspace already exists: {args.workspace}")
    args.workspace.mkdir(parents=True)

    all_frames = np.load(args.cache / "frames.npy", mmap_mode="r", allow_pickle=False)
    all_masks = np.load(args.cache / "masks.npy", mmap_mode="r", allow_pickle=False)
    n = min(args.samples, len(all_frames))
    if n < 1:
        raise SystemExit("cache contains no samples")
    frames = np.asarray(all_frames[:n])
    masks = np.asarray(all_masks[:n])
    sequential = list(range(n))
    shuffled = sequential.copy()
    random.Random(args.seed).shuffle(shuffled)
    source_hash = canonical_hash(frames, masks)

    legacy = args.workspace / "legacy-jpeg"
    save_legacy_jpeg(legacy, frames, masks)
    tar_path = args.workspace / "samples.tar"
    save_tar(tar_path, frames, masks)
    zarr_root = args.workspace / "samples.zarr"
    zarr_skip = save_zarr(zarr_root, frames, masks)

    formats: dict[str, dict[str, object]] = {
        "legacy-jpeg-q95-plus-png-mask": {
            "storage_bytes": tree_bytes(legacy),
            "lossless_roundtrip": legacy_hash(legacy, n, frames.shape[1]) == source_hash,
            "sequential_epoch": timer(
                lambda: read_legacy(legacy, sequential, frames.shape[1]),
                args.repeats,
                n,
            ),
            "shuffled_epoch": timer(
                lambda: read_legacy(legacy, shuffled, frames.shape[1]),
                args.repeats,
                n,
            ),
        },
        "contiguous-npy-mmap": {
            "storage_bytes": tree_bytes(args.cache),
            "lossless_roundtrip": npy_hash(args.cache, n) == source_hash,
            "sequential_epoch": timer(
                lambda: read_npy(args.cache, sequential), args.repeats, n
            ),
            "shuffled_epoch": timer(
                lambda: read_npy(args.cache, shuffled), args.repeats, n
            ),
        },
        "tar-npy-stream": {
            "storage_bytes": tree_bytes(tar_path),
            "lossless_roundtrip": tar_hash(tar_path) == source_hash,
            "sequential_epoch": timer(lambda: read_tar(tar_path), args.repeats, n),
            "random_access_supported": False,
        },
    }
    if zarr_skip is None:
        formats["zarr-v3-default"] = {
            "storage_bytes": tree_bytes(zarr_root),
            "lossless_roundtrip": zarr_hash(zarr_root, n) == source_hash,
            "sequential_epoch": timer(
                lambda: read_zarr(zarr_root, sequential), args.repeats, n
            ),
            "shuffled_epoch": timer(
                lambda: read_zarr(zarr_root, shuffled), args.repeats, n
            ),
        }
    else:
        formats["zarr-v3-default"] = {"skipped": zarr_skip}

    receipt = {
        "schema_version": 1,
        "kind": "data-io-benchmark",
        "identity": {
            "code_sha": args.code_sha,
            "eval_protocol_id": "gdkvm-rerelease-v1",
            "runtime_profile": "cache-format-microbenchmark",
        },
        "workload": {
            "dataset_id": args.dataset_id,
            "split_id": args.split_id,
            "sample_count": n,
            "shape": list(frames.shape),
            "repeats": args.repeats,
            "access_patterns": ["sequential_epoch", "shuffled_epoch"],
            "source_hash_sha256": source_hash,
        },
        "environment": environment_snapshot(),
        "measurement": {
            "warmup_iterations": 0,
            "measured_iterations": args.repeats,
            "formats": formats,
        },
    }
    write_receipt(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
