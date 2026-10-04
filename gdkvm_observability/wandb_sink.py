"""Optional Weights & Biases projection.

This module does not import wandb. The caller owns wandb.init() and passes the
returned Run plus wandb.Table (or a compatible factory). This keeps tests and
offline runs free from a mandatory wandb dependency.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence


class WandbSink:
    def __init__(self, run: Any, *, table_factory: Any | None = None) -> None:
        self.run = run
        self.table_factory = table_factory

    def log(self, payload: Mapping[str, Any], *, step: int) -> None:
        self.run.log(dict(payload), step=int(step))

    def log_table(
        self,
        key: str,
        *,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        step: int,
    ) -> None:
        if self.table_factory is None:
            raise RuntimeError("table_factory is required for W&B table logging")
        table = self.table_factory(columns=list(columns), data=[list(row) for row in rows])
        self.run.log({str(key): table}, step=int(step))

    def set_summary(self, payload: Mapping[str, Any]) -> None:
        for key, value in payload.items():
            self.run.summary[str(key)] = value
