#!/usr/bin/env python3
"""Compare GDKVM metric semantics with a pinned Awesome Echocardiography checkout."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_eval.metrics import (
    PROTOCOL_ID,
    bland_altman_summary,
    evaluate_segmentation_case,
    lvef_summary,
    regression_summary,
)


def load_upstream(root: Path):
    path = root / "reference" / "metrics_v1" / "reference_metrics.py"
    if not path.is_file():
        raise FileNotFoundError(f"missing upstream reference implementation: {path}")
    spec = importlib.util.spec_from_file_location("awesome_reference_metrics", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def finite_diff(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return abs(float(a) - float(b))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--random-cases", type=int, default=64)
    args = parser.parse_args()

    upstream = load_upstream(args.upstream_dir)
    rng = np.random.default_rng(args.seed)
    max_diff = {name: 0.0 for name in ("dice", "iou", "hd", "hd95", "asd")}
    statuses = 0
    cases = 0

    fixtures: list[tuple[np.ndarray, np.ndarray, tuple[float, float] | None]] = []

    empty = np.zeros((12, 15), dtype=np.uint8)
    block = empty.copy()
    block[3:9, 4:11] = 1
    shifted = np.roll(block, shift=1, axis=1)
    fixtures.extend(
        [
            (block, block, None),
            (shifted, block, None),
            (empty, empty, None),
            (empty, block, None),
            (shifted, block, (1.7, 0.6)),
        ]
    )

    for _ in range(args.random_cases):
        pred = (rng.random((24, 27)) > 0.82).astype(np.uint8)
        ref = (rng.random((24, 27)) > 0.82).astype(np.uint8)
        fixtures.append((pred, ref, (0.8, 0.5)))

    for pred, ref, spacing in fixtures:
        ours = evaluate_segmentation_case(
            pred,
            ref,
            spacing=spacing,
            distance_unit="mm" if spacing is not None else "px",
        )
        theirs = upstream.evaluate_case(
            pred,
            ref,
            spacing=spacing,
            distance_unit="mm" if spacing is not None else "px",
        )
        if ours.status != theirs.status:
            raise AssertionError(f"status mismatch: {ours.status} != {theirs.status}")
        statuses += 1
        for name in max_diff:
            diff = finite_diff(getattr(ours, name), getattr(theirs, name))
            max_diff[name] = max(max_diff[name], diff)
        cases += 1

    pred_values = np.asarray([60.0, 50.0, 70.0, 42.0, 55.5])
    ref_values = np.asarray([55.0, 52.0, 68.0, 44.0, 56.0])

    ours_reg = regression_summary(pred_values, ref_values)
    theirs_reg = upstream.regression_summary(pred_values, ref_values)
    ours_ba = bland_altman_summary(pred_values, ref_values)
    theirs_ba = upstream.bland_altman_summary(pred_values, ref_values)
    ours_lvef = lvef_summary(pred_values, ref_values)
    theirs_lvef = upstream.lvef_error_summary(pred_values, ref_values)

    clinical_diffs = {
        "mae": finite_diff(ours_reg["mae"], theirs_reg["mae"]),
        "rmse": finite_diff(ours_reg["rmse"], theirs_reg["rmse"]),
        "pearson_r": finite_diff(ours_reg["pearson_r"], theirs_reg["pearson_r"]),
        "r2": finite_diff(ours_reg["r2"], theirs_reg["r2"]),
        "bias": finite_diff(ours_ba["bias"], theirs_ba["bias"]),
        "sample_sd": finite_diff(ours_ba["sample_sd"], theirs_ba["sample_sd"]),
        "loa_low": finite_diff(ours_ba["loa_low"], theirs_ba["loa_low"]),
        "loa_high": finite_diff(ours_ba["loa_high"], theirs_ba["loa_high"]),
        "lvef_mae_pp": finite_diff(ours_lvef.mae, theirs_lvef["mae_pp"]),
        "lvef_bias_pp": finite_diff(ours_lvef.bias, theirs_lvef["bias_pp"]),
        "lvef_sd_pp": finite_diff(ours_lvef.sample_sd, theirs_lvef["sd_pp"]),
    }

    tolerance = 1e-12
    passed = (
        all(value <= tolerance for value in max_diff.values())
        and all(value <= tolerance for value in clinical_diffs.values())
    )
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "upstream_repository": "wangrui2025/awesome-echocardiography",
        "upstream_commit": args.upstream_commit,
        "upstream_reference_version": getattr(upstream, "REFERENCE_VERSION", "unknown"),
        "seed": args.seed,
        "segmentation_cases": cases,
        "status_matches": statuses,
        "tolerance": tolerance,
        "segmentation_max_abs_diff": max_diff,
        "clinical_abs_diff": clinical_diffs,
        "passed": passed,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
