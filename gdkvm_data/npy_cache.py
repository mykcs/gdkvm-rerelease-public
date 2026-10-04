"""Contiguous NumPy cache format."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


CACHE_VERSION = 1


def write_npy_cache(
    root: str | Path,
    frames: np.ndarray,
    masks: np.ndarray,
    metadata: Sequence[Mapping[str, object]],
    *,
    manifest_extra: Mapping[str, object] | None = None,
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)

    frames = np.asarray(frames)
    masks = np.asarray(masks)
    if frames.dtype != np.uint8 or masks.dtype != np.uint8:
        raise ValueError("frames and masks must be uint8")
    if frames.shape != masks.shape:
        raise ValueError("frames and masks must have the same shape")
    if frames.ndim != 4:
        raise ValueError("expected cache arrays shaped [N,T,H,W]")
    if len(metadata) != frames.shape[0]:
        raise ValueError("metadata row count must match N")

    np.save(root / "frames.npy", frames, allow_pickle=False)
    np.save(root / "masks.npy", masks, allow_pickle=False)
    with (root / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in metadata:
            handle.write(json.dumps(dict(row), sort_keys=True) + "\n")

    manifest = {
        "cache_version": CACHE_VERSION,
        "format": "contiguous-npy",
        "shape": list(frames.shape),
        "frames_dtype": str(frames.dtype),
        "masks_dtype": str(masks.dtype),
        "sample_count": int(frames.shape[0]),
    }
    if manifest_extra:
        manifest.update(dict(manifest_extra))
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return root


class NpyCacheReader:
    def __init__(self, root: str | Path, *, mmap: bool = True):
        self.root = Path(root)
        mode = "r" if mmap else None
        self.frames = np.load(self.root / "frames.npy", mmap_mode=mode, allow_pickle=False)
        self.masks = np.load(self.root / "masks.npy", mmap_mode=mode, allow_pickle=False)
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        self.metadata = [
            json.loads(line)
            for line in (self.root / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if self.frames.shape != self.masks.shape:
            raise ValueError("cache frame/mask shape mismatch")
        if len(self.metadata) != len(self.frames):
            raise ValueError("cache metadata length mismatch")

    def __len__(self) -> int:
        return int(self.frames.shape[0])

    def __getitem__(self, index: int):
        return self.frames[index], self.masks[index], self.metadata[index]

    def binary_masks(self, index: int, *, label: int = 1) -> np.ndarray:
        """Return a binary mask view for one semantic label.

        CAMUS caches preserve the official multiclass labels. GDKVM's LV task
        uses label 1, so the model-facing adapter derives that binary target at
        read time instead of discarding other structures during conversion.
        """
        return (np.asarray(self.masks[index]) == int(label)).astype(np.uint8)


def cache_nbytes(root: str | Path) -> int:
    root = Path(root)
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
