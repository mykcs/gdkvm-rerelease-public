#!/usr/bin/env python3
"""Formal single-GPU training entrypoint for the GDKVM re-release."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gdkvm_training import (
    cache_manifest_identity,
    checkpoint_identity,
    json_safe,
    memory_fraction_for_limit,
    sha256_file,
    validate_formal_cache_identity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--val-cache", type=Path, required=True)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-iterations", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--no-pretrained-backbones", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--code-sha", help="Explicit source SHA for materialized/non-git worktrees")
    return parser.parse_args()


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


def collate_gdkvm(batch):
    from torch.utils.data._utils.collate import default_collate

    tensors = {
        key: default_collate([row[key] for row in batch])
        for key in ("rgb", "ff_gt", "cls_gt", "selector")
    }
    tensors["info"] = {
        "name": [row["info"]["name"] for row in batch],
        "num_objects": default_collate(
            [row["info"]["num_objects"] for row in batch]
        ),
        "cache_index": default_collate(
            [row["info"]["cache_index"] for row in batch]
        ),
        "metadata": [row["info"]["metadata"] for row in batch],
    }
    return tensors


def make_loader(dataset, *, batch_size, shuffle, num_workers, prefetch_factor, pin_memory, seed):
    import torch
    from torch.utils.data import DataLoader

    generator = torch.Generator()
    generator.manual_seed(int(seed))
    kwargs = {
        "dataset": dataset,
        "batch_size": int(batch_size),
        "shuffle": bool(shuffle),
        "num_workers": int(num_workers),
        "pin_memory": bool(pin_memory),
        "drop_last": bool(shuffle),
        "generator": generator,
        "collate_fn": collate_gdkvm,
        "persistent_workers": int(num_workers) > 0,
    }
    if int(num_workers) > 0:
        kwargs["prefetch_factor"] = int(prefetch_factor)
    return DataLoader(**kwargs)


def main() -> int:
    args = parse_args()

    import torch
    from omegaconf import OmegaConf

    if not torch.cuda.is_available():
        raise SystemExit("formal GDKVM training currently requires CUDA")

    cfg = OmegaConf.load(args.config)
    if int(cfg.schema_version) != 1:
        raise SystemExit(f"unsupported config schema_version: {cfg.schema_version}")

    if args.max_iterations is not None:
        if args.max_iterations < 1:
            raise SystemExit("--max-iterations must be >= 1")
        cfg.training.num_iterations = int(args.max_iterations)
        cfg.main_training.num_iterations = int(args.max_iterations)
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise SystemExit("--num-workers must be >= 0")
        cfg.training.num_workers = int(args.num_workers)
    if args.no_pretrained_backbones:
        cfg.model.pretrained_backbones = False

    cfg.log_text_interval = int(cfg.logging.log_text_interval)
    cfg.debug = False

    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.resolved.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True),
        encoding="utf-8",
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output / "train.log"),
        ],
    )
    logger = logging.getLogger("gdkvm.train")

    seed = int(cfg.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda:0")
    total_memory = torch.cuda.get_device_properties(device).total_memory
    memory_fraction = memory_fraction_for_limit(
        cfg.training.get("memory_limit_gib"),
        total_memory,
    )
    if memory_fraction is not None:
        torch.cuda.set_per_process_memory_fraction(memory_fraction, device=device)

    from gdkvm_data.loader import GDKVMNpyDataset
    from gdkvm_observability import ExperimentLogger
    from model.trainer import Trainer
    from utils.unified_logger import UnifiedLogger

    formal_cache_identities = {
        "train": cache_manifest_identity(args.train_cache),
        "val": cache_manifest_identity(args.val_cache),
        "test": cache_manifest_identity(args.test_cache),
    }
    if args.max_iterations is None:
        if args.no_pretrained_backbones:
            raise SystemExit(
                "--no-pretrained-backbones is an engineering-smoke override; "
                "use a versioned config for a formal training change"
            )
        expected_samples = cfg.data.expected_samples
        expected_dataset = (
            "CAMUS"
            if "camus" in str(cfg.exp_id).lower()
            else "EchoNet-Dynamic"
        )
        for role in ("train", "val", "test"):
            validate_formal_cache_identity(
                formal_cache_identities[role],
                role=role,
                expected_dataset=expected_dataset,
                expected_samples=int(expected_samples[role]),
            )

    label = int(cfg.data.label)
    train_dataset = GDKVMNpyDataset(args.train_cache, label=label)
    val_dataset = GDKVMNpyDataset(args.val_cache, label=label)
    test_dataset = GDKVMNpyDataset(args.test_cache, label=label)

    batch_size = int(cfg.training.batch_size)
    workers = int(cfg.training.num_workers)
    prefetch = int(cfg.training.prefetch_factor)
    pin_memory = bool(cfg.training.pin_memory)
    train_loader = make_loader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
        seed=seed,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(workers, 2)),
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
        seed=seed,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=max(0, min(workers, 2)),
        prefetch_factor=prefetch,
        pin_memory=pin_memory,
        seed=seed,
    )

    local_observer = ExperimentLogger(output / "events.jsonl", rank=0)
    legacy_log = UnifiedLogger(
        str(output),
        logger,
        enabled_tb=bool(cfg.logging.tensorboard),
        enabled_wandb=False,
        git_info_enabled=False,
    )

    stage_cfg = cfg.main_training
    trainer = Trainer(
        cfg=cfg,
        stage_cfg=stage_cfg,
        log=legacy_log,
        run_path=str(output / "checkpoints"),
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        observer=local_observer,
    ).train()

    total_iterations = int(cfg.training.num_iterations)
    checkpoint_interval = int(cfg.training.checkpoint_interval)
    keep_last = int(cfg.training.keep_last_n_checkpoints)
    eval_interval = int(cfg.training.eval_interval)

    torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    data_iter = iter(train_loader)
    epoch = 0
    last_loss = None
    last_val = None
    last_test = None

    for step in range(1, total_iterations + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            data_iter = iter(train_loader)
            batch = next(data_iter)

        last_loss = float(trainer.do_pass(batch, step).detach().cpu())

        if checkpoint_interval > 0 and step % checkpoint_interval == 0:
            trainer.save_checkpoint(step, keep_last_n=keep_last)

        if (
            not args.skip_eval
            and eval_interval > 0
            and step % eval_interval == 0
        ):
            last_val = trainer.evaluate(
                val_loader, epoch=epoch + 1, run_path=str(output), it=step
            )
            last_test = trainer.test(
                test_loader, epoch=epoch + 1, run_path=str(output), it=step
            )

    final_checkpoint = trainer.save_checkpoint(
        total_iterations,
        keep_last_n=keep_last,
    )
    if not args.skip_eval:
        if last_val is None:
            last_val = trainer.evaluate(
                val_loader,
                epoch=epoch + 1,
                run_path=str(output),
                it=total_iterations,
            )
        if last_test is None:
            last_test = trainer.test(
                test_loader,
                epoch=epoch + 1,
                run_path=str(output),
                it=total_iterations,
            )

    elapsed = time.perf_counter() - start
    resolved_config = output / "config.resolved.yaml"
    receipt = {
        "schema_version": 1,
        "kind": "gdkvm-rerelease-training",
        "status": "engineering-smoke" if args.max_iterations is not None else "completed",
        "identity": {
            "code_sha": str(args.code_sha or git_sha()),
            "config_sha256": sha256_file(resolved_config),
            "eval_protocol_id": str(cfg.eval_protocol_id),
            "dataset_release_id": str(cfg.dataset_release_id),
            "runtime_profile": str(cfg.runtime.profile_id),
            "seed": seed,
        },
        "data": formal_cache_identities,
        "workload": {
            "batch_size": batch_size,
            "sequence_length": int(stage_cfg.seq_length),
            "crop_size": list(stage_cfg.crop_size),
            "num_iterations": total_iterations,
            "amp": bool(stage_cfg.amp),
            "pretrained_backbones": bool(cfg.model.pretrained_backbones),
            "memory_limit_gib": (
                float(cfg.training.memory_limit_gib)
                if cfg.training.get("memory_limit_gib") is not None
                else None
            ),
            "memory_fraction": memory_fraction,
            "augmentation": str(cfg.data.augmentation),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "cuda_runtime": getattr(torch.version, "cuda", None),
            "gpu": torch.cuda.get_device_name(device),
            "gpu_total_memory_bytes": int(total_memory),
        },
        "measurement": {
            "elapsed_s": elapsed,
            "last_train_loss": last_loss,
            "nonfinite_loss_count": int(trainer.nan_count),
            "peak_memory_allocated_bytes": int(
                torch.cuda.max_memory_allocated(device)
            ),
            "peak_memory_reserved_bytes": int(
                torch.cuda.max_memory_reserved(device)
            ),
        },
        "checkpoint": checkpoint_identity(
            final_checkpoint,
            role=(
                "engineering-smoke"
                if args.max_iterations is not None
                else "gdkvm-rerelease-candidate"
            ),
        ),
        "trainer_health_metrics": {
            "val": last_val,
            "test": last_test,
            "warning": (
                "Trainer health metrics are not the final gdkvm-rerelease-v1 "
                "publication evaluator."
            ),
        },
    }
    healthy = (
        trainer.nan_count == 0
        and last_loss is not None
        and np.isfinite(last_loss)
    )
    if args.max_iterations is not None:
        healthy = healthy and float(last_loss) > 0.0
    receipt["measurement"]["health_gate_pass"] = bool(healthy)

    safe_receipt = json_safe(receipt)
    (output / "training_receipt.json").write_text(
        json.dumps(safe_receipt, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    legacy_log.close()
    logger.info(
        "Training complete: checkpoint=%s sha256=%s health_gate=%s",
        final_checkpoint,
        receipt["checkpoint"]["sha256"],
        healthy,
    )
    print(json.dumps(safe_receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0 if healthy else 3


if __name__ == "__main__":
    raise SystemExit(main())
