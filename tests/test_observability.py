import json
import tempfile
import unittest
from pathlib import Path

from gdkvm_observability import ExperimentLogger, RunIdentity, WandbSink


class FakeRun:
    def __init__(self, *, fail_log=False):
        self.fail_log = fail_log
        self.logs = []
        self.summary = {}

    def log(self, payload, *, step):
        if self.fail_log:
            raise RuntimeError("offline")
        self.logs.append((step, dict(payload)))


class FakeTable:
    def __init__(self, *, columns, data):
        self.columns = columns
        self.data = data


def read_events(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class ObservabilityTests(unittest.TestCase):
    def test_run_identity_is_explicit_and_flattened(self):
        identity = RunIdentity(
            code_sha="abc",
            eval_protocol_id="gdkvm-rerelease-v1",
            dataset_release_id="camus-official",
            split_id="official-400-50-50",
            checkpoint_sha256="none",
            runtime_profile="eager",
            seed=7,
            world_size=1,
            compile_mode="eager",
        )
        config = identity.as_config()
        self.assertEqual(config["identity/code_sha"], "abc")
        self.assertEqual(config["identity/world_size"], 1)

    def test_one_train_event_emits_one_remote_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            run = FakeRun()
            logger = ExperimentLogger(path, projection=WandbSink(run))
            logger.log_train_step(
                step=10,
                total_loss=1.5,
                lr=2e-4,
                loss_components={"dice": 0.4, "ce": 1.1},
            )
            self.assertEqual(len(run.logs), 1)
            step, payload = run.logs[0]
            self.assertEqual(step, 10)
            self.assertEqual(payload["train/loss/total"], 1.5)
            self.assertEqual(payload["train/optim/lr"], 2e-4)
            self.assertEqual(len(read_events(path)), 1)

    def test_rank_nonzero_never_projects_remote(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rank1.jsonl"
            run = FakeRun()
            logger = ExperimentLogger(path, rank=1, projection=WandbSink(run))
            logger.log_train_step(step=1, total_loss=1.0, lr=1e-3)
            self.assertEqual(run.logs, [])
            self.assertEqual(read_events(path)[0]["rank"], 1)

    def test_remote_failure_does_not_lose_local_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            logger = ExperimentLogger(
                path,
                projection=WandbSink(FakeRun(fail_log=True)),
            )
            logger.log_train_step(step=1, total_loss=1.0, lr=1e-3)
            events = read_events(path)
            self.assertEqual(events[0]["kind"], "train_step")
            self.assertEqual(events[1]["kind"], "projection_error")

    def test_eval_summary_uses_stable_namespaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            run = FakeRun()
            logger = ExperimentLogger(path, projection=WandbSink(run))
            logger.log_eval_summary(
                "val",
                step=100,
                epoch=2,
                metrics={
                    "seg/dice": 0.91,
                    "seg/hd95_mm": 3.2,
                    "lvef/mae_pp": 4.5,
                    "count/n_valid": 50,
                },
            )
            _, payload = run.logs[0]
            self.assertEqual(payload["val/seg/dice"], 0.91)
            self.assertEqual(payload["val/lvef/mae_pp"], 4.5)
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(payload["eval_event"], "val")

    def test_numpy_metric_scalars_are_normalized_for_json(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            logger = ExperimentLogger(path)
            logger.log_eval_summary(
                "val",
                step=250,
                epoch=4,
                metrics={
                    "seg/dice": np.float32(0.9239),
                    "count/n_valid": np.int64(100),
                },
            )
            metrics = read_events(path)[0]["metrics"]
            self.assertIsInstance(metrics["val/seg/dice"], float)
            self.assertIsInstance(metrics["val/count/n_valid"], int)
            self.assertAlmostEqual(metrics["val/seg/dice"], 0.9239, places=4)
            self.assertEqual(metrics["val/count/n_valid"], 100)

    def test_eval_rows_require_pseudonymous_key_and_reject_raw_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            run = FakeRun()
            sink = WandbSink(run, table_factory=FakeTable)
            logger = ExperimentLogger(path, projection=sink)
            logger.log_eval_rows(
                "test",
                step=100,
                rows=[
                    {
                        "sample_key": "anon-001",
                        "dataset": "CAMUS",
                        "dice": 0.9,
                    }
                ],
            )
            self.assertEqual(len(run.logs), 1)
            self.assertEqual(run.logs[0][0], 100)
            table = run.logs[0][1]["test/samples"]
            self.assertEqual(table.columns, ["sample_key", "dataset", "dice"])

            with self.assertRaises(ValueError):
                logger.log_eval_rows(
                    "test",
                    step=101,
                    rows=[
                        {
                            "sample_key": "anon-002",
                            "patient_id": "patient0001",
                            "dice": 0.8,
                        }
                    ],
                )

    def test_final_summary_is_deterministic_and_records_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.jsonl"
            run = FakeRun()
            logger = ExperimentLogger(path, projection=WandbSink(run))
            logger.set_final_summary(
                "test",
                {"seg/dice": 0.92, "lvef/mae_pp": 3.5},
                selection={
                    "metric": "val/seg/dice",
                    "mode": "max",
                    "step": 500,
                    "checkpoint_sha256": "deadbeef",
                },
            )
            self.assertEqual(run.summary["summary/test/seg/dice"], 0.92)
            self.assertEqual(run.summary["selection/step"], 500)
            events = read_events(path)
            self.assertEqual(events[-1]["selection"]["mode"], "max")


if __name__ == "__main__":
    unittest.main()
