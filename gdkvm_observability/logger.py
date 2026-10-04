"""Backend-neutral experiment observability for GDKVM.

Scientific metric semantics live in the evaluation protocol. This module only
serializes already-defined events and projects rank-0 events to optional sinks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from numbers import Integral, Real
from pathlib import Path
import re
from typing import Any, Mapping, Protocol, Sequence


_KEY_RE = re.compile(r"^[a-z0-9][a-z0-9_./-]*$")
_FORBIDDEN_ROW_KEYS = {
    "filename",
    "file_name",
    "patient_id",
    "patient_name",
    "path",
    "filepath",
    "raw_id",
}


def _finite_number(value: Any) -> float | int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral):
        return int(value)
    if not isinstance(value, Real):
        raise TypeError(f"metric value must be numeric, got {type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"metric value must be finite, got {value!r}")
    return numeric


def _validate_key(key: str) -> str:
    key = str(key)
    if not _KEY_RE.fullmatch(key):
        raise ValueError(f"invalid metric key: {key!r}")
    return key


@dataclass(frozen=True)
class RunIdentity:
    code_sha: str
    eval_protocol_id: str
    dataset_release_id: str
    split_id: str
    checkpoint_sha256: str
    runtime_profile: str
    seed: int
    world_size: int
    compile_mode: str

    def __post_init__(self) -> None:
        for name in (
            "code_sha",
            "eval_protocol_id",
            "dataset_release_id",
            "split_id",
            "checkpoint_sha256",
            "runtime_profile",
            "compile_mode",
        ):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} must be non-empty")
        if self.world_size < 1:
            raise ValueError("world_size must be >= 1")

    def as_config(self) -> dict[str, Any]:
        return {f"identity/{key}": value for key, value in asdict(self).items()}


class ProjectionSink(Protocol):
    def log(self, payload: Mapping[str, Any], *, step: int) -> None: ...
    def log_table(
        self,
        key: str,
        *,
        columns: Sequence[str],
        rows: Sequence[Sequence[Any]],
        step: int,
    ) -> None: ...
    def set_summary(self, payload: Mapping[str, Any]) -> None: ...


class JsonlJournal:
    """Append-only local event journal.

    The journal is the durable fallback when optional remote observability is
    disabled or fails. Each append opens the file independently so a process
    crash does not leave a long-lived buffered file object as the only copy.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: Mapping[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(event), sort_keys=True) + "\n")
            handle.flush()


class ExperimentLogger:
    def __init__(
        self,
        journal_path: str | Path,
        *,
        rank: int = 0,
        projection: ProjectionSink | None = None,
    ) -> None:
        self.rank = int(rank)
        self.journal = JsonlJournal(journal_path)
        self.projection = projection

    @property
    def is_primary(self) -> bool:
        return self.rank == 0

    def _event(
        self,
        kind: str,
        *,
        step: int,
        metrics: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        clean_metrics = {
            _validate_key(key): _finite_number(value)
            for key, value in metrics.items()
        }
        event = {
            "kind": str(kind),
            "global_step": int(step),
            "rank": self.rank,
            "metrics": clean_metrics,
        }
        if context:
            event["context"] = dict(context)
        return event

    def _project(
        self,
        event: Mapping[str, Any],
        *,
        table: tuple[str, Sequence[str], Sequence[Sequence[Any]]] | None = None,
    ) -> None:
        if not self.is_primary or self.projection is None:
            return
        try:
            if table is None:
                payload = dict(event["metrics"])
                context = event.get("context")
                if isinstance(context, Mapping):
                    for key, value in context.items():
                        if isinstance(value, (int, float, str, bool)):
                            payload[str(key)] = value
                self.projection.log(payload, step=int(event["global_step"]))
            else:
                key, columns, rows = table
                self.projection.log_table(
                    key,
                    columns=columns,
                    rows=rows,
                    step=int(event["global_step"]),
                )
        except Exception as exc:
            self.journal.append(
                {
                    "kind": "projection_error",
                    "global_step": int(event["global_step"]),
                    "rank": self.rank,
                    "projection_error": f"{type(exc).__name__}: {exc}",
                    "source_event_kind": event["kind"],
                }
            )

    def log_train_step(
        self,
        *,
        step: int,
        total_loss: float,
        lr: float,
        loss_components: Mapping[str, float] | None = None,
        grad_norm: float | None = None,
        amp_scale: float | None = None,
        nonfinite_count: int | None = None,
        performance: Mapping[str, float] | None = None,
    ) -> None:
        metrics: dict[str, Any] = {
            "train/loss/total": total_loss,
            "train/optim/lr": lr,
        }
        for name, value in (loss_components or {}).items():
            metrics[f"train/loss/{_validate_key(name)}"] = value
        if grad_norm is not None:
            metrics["train/optim/grad_norm"] = grad_norm
        if amp_scale is not None:
            metrics["train/amp/scale"] = amp_scale
        if nonfinite_count is not None:
            metrics["train/health/nonfinite_count"] = nonfinite_count
        for name, value in (performance or {}).items():
            metrics[f"train/perf/{_validate_key(name)}"] = value

        event = self._event("train_step", step=step, metrics=metrics)
        self.journal.append(event)
        self._project(event)

    def log_eval_summary(
        self,
        stage: str,
        *,
        step: int,
        epoch: int,
        metrics: Mapping[str, float],
    ) -> None:
        stage = str(stage).lower()
        if stage not in {"val", "test"}:
            raise ValueError("evaluation stage must be 'val' or 'test'")
        namespaced = {
            f"{stage}/{_validate_key(key)}": value
            for key, value in metrics.items()
        }
        event = self._event(
            "eval_summary",
            step=step,
            metrics=namespaced,
            context={"epoch": int(epoch), "eval_event": stage},
        )
        self.journal.append(event)
        self._project(event)

    def log_eval_rows(
        self,
        stage: str,
        *,
        step: int,
        rows: Sequence[Mapping[str, Any]],
    ) -> None:
        stage = str(stage).lower()
        if stage not in {"val", "test"}:
            raise ValueError("evaluation stage must be 'val' or 'test'")
        if not rows:
            return

        columns = list(rows[0].keys())
        if "sample_key" not in columns:
            raise ValueError("evaluation rows require pseudonymous sample_key")
        forbidden = _FORBIDDEN_ROW_KEYS.intersection(columns)
        if forbidden:
            raise ValueError(
                f"evaluation rows contain forbidden raw identifiers: {sorted(forbidden)}"
            )
        for row in rows:
            if list(row.keys()) != columns:
                raise ValueError("all evaluation rows must use the same column order")

        table_rows = [[row[column] for column in columns] for row in rows]
        event = {
            "kind": "eval_rows",
            "global_step": int(step),
            "rank": self.rank,
            "stage": stage,
            "columns": columns,
            "rows": [dict(row) for row in rows],
        }
        self.journal.append(event)
        self._project(
            event,
            table=(f"{stage}/samples", columns, table_rows),
        )

    def set_final_summary(
        self,
        stage: str,
        metrics: Mapping[str, float],
        *,
        selection: Mapping[str, Any] | None = None,
    ) -> None:
        stage = str(stage).lower()
        if stage not in {"val", "test"}:
            raise ValueError("summary stage must be 'val' or 'test'")
        payload = {
            f"summary/{stage}/{_validate_key(key)}": _finite_number(value)
            for key, value in metrics.items()
        }
        event: dict[str, Any] = {
            "kind": "final_summary",
            "rank": self.rank,
            "summary": payload,
        }
        if selection:
            event["selection"] = dict(selection)
        self.journal.append(event)

        if not self.is_primary or self.projection is None:
            return
        try:
            remote = dict(payload)
            if selection:
                for key, value in selection.items():
                    remote[f"selection/{key}"] = value
            self.projection.set_summary(remote)
        except Exception as exc:
            self.journal.append(
                {
                    "kind": "projection_error",
                    "rank": self.rank,
                    "projection_error": f"{type(exc).__name__}: {exc}",
                    "source_event_kind": "final_summary",
                }
            )
