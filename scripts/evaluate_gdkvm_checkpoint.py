#!/usr/bin/env python3
"""Evaluate one newly trained GDKVM checkpoint under the re-release metric contract."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_eval import (
    CLINICAL_METHOD_ID,
    PROTOCOL_ID,
    area_length_lvef_percent,
    aggregate_segmentation,
    evaluate_segmentation_case,
    lvef_summary,
    patient_bootstrap_mean_ci,
)
from gdkvm_training import (
    cache_manifest_identity,
    checkpoint_identity,
    json_safe,
    validate_formal_cache_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--code-sha", help="Explicit evaluator source SHA")
    return parser.parse_args()


def pseudonymous_key(value: str) -> str:
    return sha256(str(value).encode("utf-8")).hexdigest()[:16]


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
    except Exception:
        return "unknown"


def expected_dataset_and_count(cfg, split: str) -> tuple[str, int]:
    dataset = "CAMUS" if "camus" in str(cfg.exp_id).lower() else "EchoNet-Dynamic"
    return dataset, int(cfg.data.expected_samples[split])


def main() -> int:
    args = parse_args()
    import torch
    from omegaconf import OmegaConf
    from torch.utils.data._utils.collate import default_collate

    from gdkvm_data.loader import GDKVMNpyDataset
    from gdkvm_runtime import prepare_model_runtime, resolve_runtime_policy
    from model.gdkvm01 import GDKVM

    cfg = OmegaConf.load(args.config)
    if str(cfg.eval_protocol_id) != PROTOCOL_ID:
        raise SystemExit(
            f"config eval_protocol_id {cfg.eval_protocol_id!r} != {PROTOCOL_ID!r}"
        )

    cache_identity = cache_manifest_identity(args.cache)
    engineering_subset = args.max_samples is not None
    if engineering_subset:
        if args.max_samples < 1:
            raise SystemExit("--max-samples must be >= 1")
    else:
        dataset_name, expected_count = expected_dataset_and_count(cfg, args.split)
        validate_formal_cache_identity(
            cache_identity,
            role=args.split,
            expected_dataset=dataset_name,
            expected_samples=expected_count,
        )

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device)

    torch.manual_seed(int(cfg.seed))
    if device.type == "cuda":
        torch.cuda.manual_seed_all(int(cfg.seed))

    runtime_policy = resolve_runtime_policy(cfg)
    if runtime_policy.compile_enabled:
        raise SystemExit(
            "checkpoint publication evaluation uses the eager reference path; "
            "use a config with runtime.compile.enabled=false"
        )

    model = GDKVM(
        model_type=cfg.model.get("model_type", "base"),
        image_encoder_type=cfg.model.get("image_encoder_type", "resnet50"),
        mask_encoder_type=cfg.model.get("mask_encoder_type", "resnet18"),
        use_channels_last=runtime_policy.channels_last,
        pretrained_backbones=False,
    )
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    load_result = model.load_state_dict(state, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise RuntimeError(
            f"checkpoint strict load mismatch: missing={load_result.missing_keys}, "
            f"unexpected={load_result.unexpected_keys}"
        )
    model = prepare_model_runtime(model.to(device), torch, runtime_policy).eval()

    dataset = GDKVMNpyDataset(args.cache, label=int(cfg.data.label))
    limit = len(dataset) if args.max_samples is None else min(len(dataset), args.max_samples)

    rows = []
    cases = []
    patient_ids = []
    dice_values = []
    hd95_values = []
    asd_values = []
    lvef_pred_percent = []
    lvef_ref_percent = []
    lvef_views = []
    lvef_patients = []
    clinical_rows = []

    with torch.inference_mode():
        for index in range(limit):
            sample = dataset[index]
            metadata = sample["info"]["metadata"]
            batch = default_collate([sample])
            raw_num = batch["info"]["num_objects"]
            if isinstance(raw_num, torch.Tensor):
                batch["info"]["num_objects"] = [
                    int(value) for value in raw_num.detach().cpu().tolist()
                ]
            for key, value in list(batch.items()):
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(device)

            output = model(batch)
            frame_count = int(batch["rgb"].shape[1])
            endpoint_pred = {}
            endpoint_ref = {}
            for phase_name, frame_index in (("first", 0), ("last", frame_count - 1)):
                key = f"masks_{frame_index}"
                if key not in output:
                    raise RuntimeError(f"model output missing {key}")
                pred = output[key]
                if pred.shape[1] > 1:
                    pred = pred[:, :1]
                pred_mask = (pred[0, 0] > 0.5).detach().cpu().numpy()
                ref_mask = (
                    batch["cls_gt"][0, frame_index, 0]
                    .detach()
                    .cpu()
                    .numpy()
                    > 0
                )
                endpoint_pred[phase_name] = pred_mask
                endpoint_ref[phase_name] = ref_mask

                spacing = metadata.get("resized_spacing_axis01")
                unit = "mm" if spacing is not None else "px"
                result = evaluate_segmentation_case(
                    pred_mask,
                    ref_mask,
                    spacing=spacing,
                    distance_unit=unit,
                )
                cases.append(result)

                patient_id = str(
                    metadata.get("patient_id")
                    or metadata.get("sample_id")
                    or sample["info"]["name"]
                )
                patient_ids.append(patient_id)
                dice_values.append(result.dice)
                hd95_values.append(result.hd95)
                asd_values.append(result.asd)
                rows.append(
                    {
                        "sample_key": pseudonymous_key(sample["info"]["name"]),
                        "phase": phase_name,
                        "status": result.status,
                        "distance_unit": result.distance_unit,
                        "dice": result.dice,
                        "iou": result.iou,
                        "hd": result.hd,
                        "hd95": result.hd95,
                        "asd": result.asd,
                    }
                )

                if phase_name == "last":
                    pred_ef = area_length_lvef_percent(
                        endpoint_pred["first"], endpoint_pred["last"]
                    )
                    ref_ef = area_length_lvef_percent(
                        endpoint_ref["first"], endpoint_ref["last"]
                    )
                    view = str(metadata.get("view") or "unknown")
                    lvef_pred_percent.append(pred_ef)
                    lvef_ref_percent.append(ref_ef)
                    lvef_views.append(view)
                    lvef_patients.append(patient_id)
                    clinical_rows.append(
                        {
                            "sample_key": pseudonymous_key(sample["info"]["name"]),
                            "patient_key": pseudonymous_key(patient_id),
                            "view": view,
                            "prediction_percent": pred_ef,
                            "reference_percent": ref_ef,
                        }
                    )

    aggregate = aggregate_segmentation(cases)
    bootstrap = {
        "dice": patient_bootstrap_mean_ci(
            dice_values, patient_ids, n_resamples=2000, seed=int(cfg.seed)
        ),
        "hd95": patient_bootstrap_mean_ci(
            hd95_values, patient_ids, n_resamples=2000, seed=int(cfg.seed) + 1
        ),
        "asd": patient_bootstrap_mean_ci(
            asd_values, patient_ids, n_resamples=2000, seed=int(cfg.seed) + 2
        ),
    }

    clinical_overall = lvef_summary(
        lvef_pred_percent, lvef_ref_percent
    ).to_dict()
    clinical_by_view = {}
    for view in sorted(set(lvef_views)):
        selected = [i for i, item in enumerate(lvef_views) if item == view]
        clinical_by_view[view] = lvef_summary(
            [lvef_pred_percent[i] for i in selected],
            [lvef_ref_percent[i] for i in selected],
        ).to_dict()

    ckpt = checkpoint_identity(
        args.checkpoint,
        role="engineering-eval" if engineering_subset else "gdkvm-rerelease-candidate",
    )
    receipt = {
        "schema_version": 1,
        "kind": "gdkvm-checkpoint-evaluation",
        "status": "engineering-smoke" if engineering_subset else "completed",
        "identity": {
            "eval_protocol_id": PROTOCOL_ID,
            "dataset_release_id": str(cfg.dataset_release_id),
            "runtime_profile": str(cfg.runtime.profile_id),
            "seed": int(cfg.seed),
            "evaluator_code_sha": str(args.code_sha or git_sha()),
        },
        "checkpoint": ckpt,
        "data": cache_identity,
        "workload": {
            "split": args.split,
            "evaluated_samples": limit,
            "atomic_cases": len(cases),
            "label": int(cfg.data.label),
            "device": str(device),
        },
        "segmentation": {
            "aggregate": aggregate,
            "patient_bootstrap_95ci": bootstrap,
            "rows": rows,
        },
        "clinical": {
            "upstream_method": {
                "id": CLINICAL_METHOD_ID,
                "endpoint_binding": "CAMUS half-sequence first=ED, last=ES",
                "volume_formula": "8*A^2/(3*pi*L)",
                "area_unit": "resized-pixel^2",
                "length_unit": "resized-pixel",
                "ef_unit": "percent",
                "note": (
                    "Single-view segmentation-derived area-length surrogate; "
                    "not biplane Simpson and not dataset-provided clinical LVEF."
                ),
            },
            "overall_view_samples": clinical_overall,
            "by_view": clinical_by_view,
            "rows": clinical_rows,
        },
        "non_claims": [
            "Area-length LVEF is a named single-view surrogate, not biplane Simpson",
            "engineering subsets are never publication results",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    safe_receipt = json_safe(receipt)
    args.output.write_text(
        json.dumps(safe_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(safe_receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
