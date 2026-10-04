"""Independent evaluator for the GDKVM re-release protocol.

The implementation follows the mathematical contract in evaluation/PROTOCOL.md.
It intentionally does not vendor the Awesome Echocardiography reference source.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from typing import Iterable, Sequence

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


PROTOCOL_ID = "gdkvm-rerelease-v1"


@dataclass(frozen=True)
class SegmentationCase:
    dice: float
    iou: float
    hd: float
    hd95: float
    asd: float
    status: str
    spacing_y: float
    spacing_x: float
    distance_unit: str

    def to_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class ClinicalSummary:
    n: int
    mae: float
    rmse: float
    pearson_r: float
    r2: float
    bias: float
    sample_sd: float
    loa_low: float
    loa_high: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _mask2d(mask, name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    return arr.astype(bool, copy=False)


def _spacing2d(spacing: Sequence[float] | float | None) -> tuple[float, float]:
    if spacing is None:
        return (1.0, 1.0)
    if isinstance(spacing, (int, float)):
        values = (float(spacing), float(spacing))
    else:
        if len(spacing) != 2:
            raise ValueError("spacing must be a scalar or (y, x)")
        values = (float(spacing[0]), float(spacing[1]))
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("spacing values must be finite and > 0")
    return values


def _surface(mask: np.ndarray) -> np.ndarray:
    return binary_erosion(mask) ^ mask


def _directed_surface_distances(
    source: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float],
) -> np.ndarray:
    source_surface = _surface(source)
    target_surface = _surface(target)
    if not source_surface.any():
        return np.empty((0,), dtype=np.float64)
    if not target_surface.any():
        return np.full(int(source_surface.sum()), np.inf, dtype=np.float64)
    distance_map = distance_transform_edt(~target_surface, sampling=spacing)
    return np.asarray(distance_map[source_surface], dtype=np.float64)


def _field_of_view_penalty(
    shape: tuple[int, int],
    spacing: tuple[float, float],
) -> float:
    height, width = shape
    if height < 1 or width < 1:
        raise ValueError("mask dimensions must be >= 1")
    sy, sx = spacing
    return sqrt(((height - 1) * sy) ** 2 + ((width - 1) * sx) ** 2)


def evaluate_segmentation_case(
    pred,
    reference,
    *,
    spacing: Sequence[float] | float | None = None,
    distance_unit: str = "px",
) -> SegmentationCase:
    """Evaluate one atomic binary-mask pair."""
    pred_mask = _mask2d(pred, "pred")
    ref_mask = _mask2d(reference, "reference")
    if pred_mask.shape != ref_mask.shape:
        raise ValueError("pred and reference must have the same shape")

    sy, sx = _spacing2d(spacing)
    pred_nonempty = bool(pred_mask.any())
    ref_nonempty = bool(ref_mask.any())

    if not pred_nonempty and not ref_nonempty:
        nan = float("nan")
        return SegmentationCase(
            nan, nan, nan, nan, nan, "both_empty", sy, sx, distance_unit
        )

    if pred_nonempty != ref_nonempty:
        penalty = _field_of_view_penalty(pred_mask.shape, (sy, sx))
        return SegmentationCase(
            0.0, 0.0, penalty, penalty, penalty, "one_empty", sy, sx, distance_unit
        )

    intersection = int(np.logical_and(pred_mask, ref_mask).sum())
    pred_size = int(pred_mask.sum())
    ref_size = int(ref_mask.sum())
    union = int(np.logical_or(pred_mask, ref_mask).sum())

    dice = 2.0 * intersection / (pred_size + ref_size)
    iou = intersection / union

    pred_to_ref = _directed_surface_distances(pred_mask, ref_mask, (sy, sx))
    ref_to_pred = _directed_surface_distances(ref_mask, pred_mask, (sy, sx))

    hd = float(max(pred_to_ref.max(), ref_to_pred.max()))
    hd95 = float(
        max(
            np.quantile(pred_to_ref, 0.95, method="linear"),
            np.quantile(ref_to_pred, 0.95, method="linear"),
        )
    )
    asd = float(np.concatenate((pred_to_ref, ref_to_pred)).mean())

    return SegmentationCase(
        float(dice),
        float(iou),
        hd,
        hd95,
        asd,
        "ok",
        sy,
        sx,
        distance_unit,
    )


def aggregate_segmentation(
    cases: Iterable[SegmentationCase],
) -> dict[str, float | int]:
    """Macro-average atomic items while preserving explicit empty counts."""
    rows = list(cases)
    valid = [row for row in rows if row.status != "both_empty"]
    counts = {
        "n_total": len(rows),
        "n_valid": len(valid),
        "n_one_empty": sum(row.status == "one_empty" for row in rows),
        "n_both_empty": sum(row.status == "both_empty" for row in rows),
    }
    if not valid:
        return {
            **counts,
            "dice_mean": float("nan"),
            "iou_mean": float("nan"),
            "hd_mean": float("nan"),
            "hd95_mean": float("nan"),
            "asd_mean": float("nan"),
        }

    return {
        **counts,
        "dice_mean": float(np.mean([row.dice for row in valid])),
        "iou_mean": float(np.mean([row.iou for row in valid])),
        "hd_mean": float(np.mean([row.hd for row in valid])),
        "hd95_mean": float(np.mean([row.hd95 for row in valid])),
        "asd_mean": float(np.mean([row.asd for row in valid])),
    }


def _paired_finite(
    pred: Sequence[float],
    reference: Sequence[float],
) -> tuple[np.ndarray, np.ndarray]:
    pred_arr = np.asarray(pred, dtype=np.float64)
    ref_arr = np.asarray(reference, dtype=np.float64)
    if pred_arr.ndim != 1 or ref_arr.ndim != 1:
        raise ValueError("pred and reference must be 1-D")
    if pred_arr.shape != ref_arr.shape:
        raise ValueError("pred and reference must have the same shape")
    finite = np.isfinite(pred_arr) & np.isfinite(ref_arr)
    return pred_arr[finite], ref_arr[finite]


def regression_summary(
    pred: Sequence[float],
    reference: Sequence[float],
) -> dict[str, float | int]:
    pred_arr, ref_arr = _paired_finite(pred, reference)
    n = len(pred_arr)
    if n == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "pearson_r": float("nan"),
            "r2": float("nan"),
        }

    error = pred_arr - ref_arr
    pred_sd = float(np.std(pred_arr))
    ref_sd = float(np.std(ref_arr))
    pearson = (
        float(np.corrcoef(pred_arr, ref_arr)[0, 1])
        if n >= 2 and pred_sd > 0 and ref_sd > 0
        else float("nan")
    )
    sst = float(np.sum((ref_arr - ref_arr.mean()) ** 2))
    r2 = (
        float(1.0 - np.sum(error**2) / sst)
        if sst > 0
        else float("nan")
    )
    return {
        "n": int(n),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson_r": pearson,
        "r2": r2,
    }


def bland_altman_summary(
    pred: Sequence[float],
    reference: Sequence[float],
) -> dict[str, float | int | str]:
    """Use prediction-reference and sample SD (ddof=1)."""
    pred_arr, ref_arr = _paired_finite(pred, reference)
    n = len(pred_arr)
    if n == 0:
        return {
            "n": 0,
            "difference": "prediction-reference",
            "bias": float("nan"),
            "sample_sd": float("nan"),
            "loa_low": float("nan"),
            "loa_high": float("nan"),
        }
    diff = pred_arr - ref_arr
    bias = float(diff.mean())
    if n < 2:
        sample_sd = loa_low = loa_high = float("nan")
    else:
        sample_sd = float(np.std(diff, ddof=1))
        loa_low = float(bias - 1.96 * sample_sd)
        loa_high = float(bias + 1.96 * sample_sd)
    return {
        "n": int(n),
        "difference": "prediction-reference",
        "bias": bias,
        "sample_sd": sample_sd,
        "loa_low": loa_low,
        "loa_high": loa_high,
    }


def lvef_summary(
    pred_percent: Sequence[float],
    reference_percent: Sequence[float],
) -> ClinicalSummary:
    pred_arr, ref_arr = _paired_finite(pred_percent, reference_percent)
    reg = regression_summary(pred_arr, ref_arr)
    agreement = bland_altman_summary(pred_arr, ref_arr)
    return ClinicalSummary(
        n=int(reg["n"]),
        mae=float(reg["mae"]),
        rmse=float(reg["rmse"]),
        pearson_r=float(reg["pearson_r"]),
        r2=float(reg["r2"]),
        bias=float(agreement["bias"]),
        sample_sd=float(agreement["sample_sd"]),
        loa_low=float(agreement["loa_low"]),
        loa_high=float(agreement["loa_high"]),
    )


def patient_bootstrap_mean_ci(
    values: Sequence[float],
    patient_ids: Sequence[str],
    *,
    confidence: float = 0.95,
    n_resamples: int = 2000,
    seed: int = 0,
) -> dict[str, float | int]:
    """Bootstrap the mean by resampling patients, never pixels/frames.

    Multiple rows for one patient are kept together. Each resampled patient's
    contribution is its within-patient mean, so patients receive equal weight.
    """
    value_arr = np.asarray(values, dtype=np.float64)
    patient_arr = np.asarray(patient_ids, dtype=object)
    if value_arr.ndim != 1 or patient_arr.ndim != 1:
        raise ValueError("values and patient_ids must be 1-D")
    if value_arr.shape != patient_arr.shape:
        raise ValueError("values and patient_ids must have the same shape")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n_resamples < 1:
        raise ValueError("n_resamples must be >= 1")

    finite = np.isfinite(value_arr)
    value_arr = value_arr[finite]
    patient_arr = patient_arr[finite]
    if len(value_arr) == 0:
        return {
            "n_patients": 0,
            "mean": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
        }

    unique_patients = np.unique(patient_arr)
    patient_means = np.asarray(
        [value_arr[patient_arr == pid].mean() for pid in unique_patients],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    samples = rng.choice(
        patient_means,
        size=(n_resamples, len(patient_means)),
        replace=True,
    ).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "n_patients": int(len(patient_means)),
        "mean": float(patient_means.mean()),
        "ci_low": float(np.quantile(samples, alpha, method="linear")),
        "ci_high": float(np.quantile(samples, 1.0 - alpha, method="linear")),
    }
