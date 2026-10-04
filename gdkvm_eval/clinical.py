"""Checkpoint-bound clinical reconstruction helpers.

This module binds one explicit upstream LVEF method for the GDKVM re-release
without changing the metric semantics in gdkvm_eval.metrics.
"""

from __future__ import annotations

from math import pi

import numpy as np
from skimage.measure import regionprops


CLINICAL_METHOD_ID = "single-view-area-length-v1"


def area_length_volume(mask) -> float:
    """Return the historical GDKVM single-view area-length volume surrogate.

    The binary mask is passed directly to regionprops as one labelled
    foreground region, matching the Trainer health-metric implementation.
    """
    arr = np.asarray(mask).astype(bool, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {arr.shape}")
    if not arr.any():
        return 0.0

    props = regionprops(arr.astype(np.uint8))
    if not props:
        return 0.0
    region = max(props, key=lambda item: item.area)
    area = float(region.area)
    length = float(region.major_axis_length)
    if length <= 0.0 or not np.isfinite(length):
        return 0.0
    return float((8.0 * area**2) / (3.0 * pi * length))


def area_length_lvef_percent(ed_mask, es_mask) -> float:
    """Compute LVEF (%) from endpoint masks using the area-length surrogate."""
    edv = area_length_volume(ed_mask)
    esv = area_length_volume(es_mask)
    if edv <= 0.0 or not np.isfinite(edv) or not np.isfinite(esv):
        return float("nan")
    return float(100.0 * (edv - esv) / edv)
