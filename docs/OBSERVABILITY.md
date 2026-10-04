# GDKVM observability

This package owns experiment-event serialization and optional dashboard projection.

It does **not** own metric formulas. Metric semantics are defined by
`evaluation/PROTOCOL.md`.

## Trainer boundary

Core training/evaluation code emits one structured event per logical event:

- `log_train_step(...)`
- `log_eval_summary(...)`
- `log_eval_rows(...)`
- `set_final_summary(...)`

The logger always writes an append-only local JSONL journal. Rank 0 may also
project the same event to W&B through `WandbSink`.

Trainer code must not call `wandb.log()` or `wandb.watch()` directly.

## W&B initialization

The W&B SDK stays optional. The entrypoint owns SDK initialization:

```python
import wandb

from gdkvm_observability import (
    ExperimentLogger,
    RunIdentity,
    WandbSink,
    build_wandb_init_kwargs,
)

identity = RunIdentity(
    code_sha=code_sha,
    eval_protocol_id="gdkvm-rerelease-v1",
    dataset_release_id=dataset_release_id,
    split_id=split_id,
    checkpoint_sha256=checkpoint_sha256,
    runtime_profile=runtime_profile,
    seed=seed,
    world_size=world_size,
    compile_mode=compile_mode,
)

kwargs = build_wandb_init_kwargs(
    identity,
    project="GDKVM-rerelease",
    config=plain_hydra_config,
    tags=["rerelease-v1", runtime_profile],
)

run = wandb.init(**kwargs)
logger = ExperimentLogger(
    local_receipt_path,
    rank=rank,
    projection=WandbSink(run, table_factory=wandb.Table),
)
```

W&B's current Hydra guidance recommends converting OmegaConf/DictConfig to a
plain container before passing it as run config.

## Privacy

Sample-level tables require a pseudonymous `sample_key`. Raw patient IDs,
filenames, filesystem paths, and other direct source identifiers are rejected
by the logger's table boundary.

This is intentionally stricter than W&B's technical capabilities: Tables can
carry rich media, but dataset terms and privacy rules decide what may be
uploaded.

## Failure behavior

A remote projection exception never deletes or rolls back the local event.
The journal records a `projection_error` event and training may continue.

No W&B API key belongs in the repository.

## Model watch

`wandb.watch(..., log="all")` is not part of the default contract. If a later
Trainer integration adds model watching for diagnosis, it must be an explicit
debug profile and its overhead must be measured.
