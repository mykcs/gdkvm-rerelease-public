#!/usr/bin/env python3
"""Benchmark PyTorch DataLoader throughput for NPY cache vs legacy image folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Sequence

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_bench import environment_snapshot, write_receipt


class NpyDataset:
    def __init__(self, root: Path):
        self.frames = np.load(root / "frames.npy", mmap_mode="r", allow_pickle=False)
        self.masks = np.load(root / "masks.npy", mmap_mode="r", allow_pickle=False)

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int):
        # Return writable arrays so default_collate can safely turn them into tensors.
        return (
            np.array(self.frames[index], copy=True),
            np.array(self.masks[index], copy=True),
        )


class LegacyImageDataset:
    def __init__(self, root: Path, sample_count: int, frames_per_sample: int):
        self.root = root
        self.sample_count = sample_count
        self.frames_per_sample = frames_per_sample

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int):
        sample_root = self.root / f"{index:05d}"
        frames = []
        masks = []
        for frame in range(self.frames_per_sample):
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
        return np.stack(frames), np.stack(masks)


def run_epoch(loader) -> tuple[float, int, int]:
    start = time.perf_counter()
    checksum = 0
    samples = 0
    for frames, masks in loader:
        samples += int(frames.shape[0])
        checksum ^= int(frames[:, 0, 0, 0].sum().item())
        checksum ^= int(masks[:, 0, 0, 0].sum().item())
    return time.perf_counter() - start, samples, checksum


def benchmark_dataset(
    dataset,
    *,
    workers: int,
    batch_size: int,
    warmup_epochs: int,
    measured_epochs: int,
    seed: int,
) -> dict[str, object]:
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=False,
        drop_last=False,
        generator=generator,
    )
    for _ in range(warmup_epochs):
        run_epoch(loader)

    times: list[float] = []
    checksum = 0
    count = 0
    for _ in range(measured_epochs):
        elapsed, count, epoch_checksum = run_epoch(loader)
        times.append(elapsed)
        checksum ^= epoch_checksum

    mean_s = statistics.fmean(times)
    return {
        "workers": workers,
        "batch_size": batch_size,
        "warmup_epochs": warmup_epochs,
        "measured_epochs": measured_epochs,
        "sample_count": count,
        "mean_s": mean_s,
        "median_s": statistics.median(times),
        "min_s": min(times),
        "max_s": max(times),
        "samples_per_s": count / max(mean_s, 1e-12),
        "checksum": checksum,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--legacy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-sha", required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--split-id", required=True)
    parser.add_argument("--workers", default="0,2,4,8")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--measured-epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    worker_values: Sequence[int] = [
        int(value) for value in args.workers.split(",") if value.strip()
    ]
    if any(value < 0 for value in worker_values):
        raise SystemExit("workers must be >= 0")

    npy_dataset = NpyDataset(args.cache)
    frames_per_sample = int(npy_dataset.frames.shape[1])
    legacy_dataset = LegacyImageDataset(
        args.legacy,
        sample_count=len(npy_dataset),
        frames_per_sample=frames_per_sample,
    )

    results: dict[str, dict[str, object]] = {"npy": {}, "legacy_jpeg": {}}
    for workers in worker_values:
        results["npy"][str(workers)] = benchmark_dataset(
            npy_dataset,
            workers=workers,
            batch_size=args.batch_size,
            warmup_epochs=args.warmup_epochs,
            measured_epochs=args.measured_epochs,
            seed=args.seed,
        )
        results["legacy_jpeg"][str(workers)] = benchmark_dataset(
            legacy_dataset,
            workers=workers,
            batch_size=args.batch_size,
            warmup_epochs=args.warmup_epochs,
            measured_epochs=args.measured_epochs,
            seed=args.seed,
        )

    receipt = {
        "schema_version": 1,
        "kind": "data-io-benchmark",
        "identity": {
            "code_sha": args.code_sha,
            "eval_protocol_id": "gdkvm-rerelease-v1",
            "runtime_profile": "pytorch-dataloader-cpu",
        },
        "workload": {
            "dataset_id": args.dataset_id,
            "split_id": args.split_id,
            "sample_count": len(npy_dataset),
            "sequence_length": frames_per_sample,
            "input_resolution": list(npy_dataset.frames.shape[-2:]),
            "batch_size": args.batch_size,
            "shuffle": True,
            "pin_memory": False,
            "worker_values": list(worker_values),
        },
        "environment": environment_snapshot(),
        "measurement": {
            "warmup_iterations": args.warmup_epochs,
            "measured_iterations": args.measured_epochs,
            "results": results,
        },
    }
    write_receipt(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
