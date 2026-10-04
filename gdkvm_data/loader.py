"""Model-facing adapter for the canonical contiguous NPY cache.

The adapter intentionally depends only on NumPy. PyTorch's default_collate can
turn the returned arrays into tensors, so this module remains usable in Gate A
CPU/data validation without forcing a torch installation.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .npy_cache import NpyCacheReader


class GDKVMNpyDataset:
    """Expose one cached sequence in the structure expected by GDKVM training.

    The cache preserves all official labels. The model-facing adapter derives a
    binary target for the requested semantic label (LV endocardium defaults to
    label 1 for CAMUS). EchoNet caches are already binary.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        label: int = 1,
        max_num_obj: int = 1,
        normalize_images: bool = True,
        mmap: bool = True,
    ) -> None:
        if max_num_obj != 1:
            raise ValueError("current GDKVM re-release adapter supports max_num_obj=1")
        self.cache = NpyCacheReader(root, mmap=mmap)
        self.label = int(label)
        self.max_num_obj = int(max_num_obj)
        self.normalize_images = bool(normalize_images)

    def __len__(self) -> int:
        return len(self.cache)

    def __getitem__(self, index: int) -> dict[str, Any]:
        frames, _, metadata = self.cache[index]
        frames = np.asarray(frames)
        masks = self.cache.binary_masks(index, label=self.label)

        if frames.ndim != 3 or masks.ndim != 3:
            raise ValueError("cached sample must be shaped [T,H,W]")
        if frames.shape != masks.shape:
            raise ValueError("cached frame/mask sample shape mismatch")

        if self.normalize_images:
            rgb = frames.astype(np.float32, copy=False) / 255.0
        else:
            rgb = frames.astype(np.uint8, copy=False)
        rgb = rgb[:, None, :, :]

        cls_gt = masks.astype(np.int64, copy=False)[:, None, :, :]
        height, width = masks.shape[-2:]
        first_frame_gt = np.zeros(
            (1, self.max_num_obj, height, width), dtype=np.int64
        )
        selector = np.zeros((self.max_num_obj,), dtype=np.float32)

        present = bool(masks[0].any())
        if present:
            first_frame_gt[0, 0] = masks[0]
            selector[0] = 1.0

        info = {
            "name": str(metadata.get("sample_id", index)),
            "num_objects": 1 if present else 0,
            "cache_index": int(index),
            "metadata": dict(metadata),
        }
        return {
            "rgb": rgb,
            "ff_gt": first_frame_gt,
            "cls_gt": cls_gt,
            "selector": selector,
            "info": info,
        }
