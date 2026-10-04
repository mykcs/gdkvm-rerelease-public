"""Helpers for constructing a W&B run without importing the W&B SDK.

The training entrypoint may call:

    kwargs = build_wandb_init_kwargs(...)
    run = wandb.init(**kwargs)
    logger = ExperimentLogger(..., projection=WandbSink(run, table_factory=wandb.Table))

Keeping this module backend-agnostic makes the core observability contract
testable without network access or a wandb installation.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .logger import RunIdentity


def build_wandb_init_kwargs(
    identity: RunIdentity,
    *,
    project: str,
    config: Mapping[str, Any] | None = None,
    entity: str | None = None,
    name: str | None = None,
    tags: Sequence[str] = (),
) -> dict[str, Any]:
    if not str(project).strip():
        raise ValueError("project must be non-empty")

    run_config = dict(config or {})
    collisions = set(run_config).intersection(identity.as_config())
    if collisions:
        raise ValueError(
            "base config must not overwrite identity keys: "
            + ", ".join(sorted(collisions))
        )
    run_config.update(identity.as_config())

    kwargs: dict[str, Any] = {
        "project": str(project),
        "config": run_config,
        "tags": list(dict.fromkeys(str(tag) for tag in tags if str(tag).strip())),
    }
    if entity:
        kwargs["entity"] = str(entity)
    if name:
        kwargs["name"] = str(name)
    return kwargs


def default_run_name(
    *,
    dataset: str,
    model: str,
    runtime: str,
    seed: int,
    code_sha: str,
) -> str:
    short_sha = str(code_sha)[:8]
    return f"{dataset}-{model}-{runtime}-s{int(seed)}-{short_sha}"
