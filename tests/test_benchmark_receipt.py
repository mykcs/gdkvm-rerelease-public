import json
import tempfile
import unittest
from pathlib import Path

from gdkvm_bench.receipt import (
    PhaseRecorder,
    environment_snapshot,
    percentile,
    sha256_file,
    validate_receipt,
    write_receipt,
)


class BenchmarkReceiptTests(unittest.TestCase):
    def test_environment_snapshot_makes_nvidia_availability_explicit(self):
        snapshot = environment_snapshot()
        self.assertIn("nvidia_smi_available", snapshot)
        self.assertIn("tempdir", snapshot)
        if not snapshot["nvidia_smi_available"]:
            self.assertIn("nvidia_smi_error", snapshot)

    def test_percentile_linear_interpolation(self):
        self.assertAlmostEqual(percentile([0.0, 10.0], 0.95), 9.5)
        self.assertAlmostEqual(percentile([3.0], 0.25), 3.0)

    def test_phase_recorder_emits_summary(self):
        recorder = PhaseRecorder()
        for _ in range(3):
            with recorder.phase("step"):
                sum(range(1000))
        summary = recorder.summary()["step"]
        self.assertEqual(summary["n"], 3)
        self.assertGreaterEqual(summary["max_s"], summary["min_s"])
        self.assertGreaterEqual(summary["p95_s"], summary["median_s"])

    def test_phase_recorder_synchronizes_before_and_after_timing(self):
        calls = []
        recorder = PhaseRecorder(synchronize=lambda: calls.append("sync"))
        with recorder.phase("gpu_like_step"):
            pass
        self.assertEqual(calls, ["sync", "sync"])

    def test_receipt_validation_rejects_missing_identity(self):
        with self.assertRaises(ValueError):
            validate_receipt(
                {
                    "schema_version": 1,
                    "kind": "test",
                    "identity": {},
                    "workload": {},
                    "environment": {},
                    "measurement": {
                        "warmup_iterations": 0,
                        "measured_iterations": 1,
                    },
                }
            )

    def test_performance_receipt_requires_matched_context_fields(self):
        receipt = {
            "schema_version": 1,
            "kind": "performance",
            "identity": {
                "code_sha": "abc",
                "eval_protocol_id": "gdkvm-rerelease-v1",
                "runtime_profile": "eager",
            },
            "workload": {},
            "environment": {},
            "measurement": {
                "warmup_iterations": 2,
                "measured_iterations": 10,
            },
        }
        with self.assertRaises(ValueError):
            validate_receipt(receipt)

    def test_complete_performance_receipt_is_valid(self):
        receipt = {
            "schema_version": 1,
            "kind": "performance",
            "identity": {
                "code_sha": "abc",
                "eval_protocol_id": "gdkvm-rerelease-v1",
                "runtime_profile": "eager",
                "checkpoint_sha256": "deadbeef",
                "dataset_id": "synthetic-v1",
                "split_id": "fixture",
                "compile_mode": "eager",
            },
            "workload": {
                "input_resolution": [128, 128],
                "sequence_length": 10,
                "batch_size": 2,
                "dtype": "float32",
                "amp": False,
                "world_size": 1,
                "logging_profile": "minimal",
                "data_cache_format": "synthetic",
            },
            "environment": {},
            "measurement": {
                "warmup_iterations": 2,
                "measured_iterations": 10,
                "phases": {},
            },
        }
        validate_receipt(receipt)

    def test_write_receipt_is_sorted_and_valid(self):
        receipt = {
            "schema_version": 1,
            "kind": "test",
            "identity": {
                "code_sha": "abc",
                "eval_protocol_id": "gdkvm-rerelease-v1",
                "runtime_profile": "unit-test",
            },
            "workload": {},
            "environment": {},
            "measurement": {
                "warmup_iterations": 0,
                "measured_iterations": 1,
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "receipt.json"
            write_receipt(path, receipt)
            loaded = json.loads(path.read_text())
            self.assertEqual(loaded, receipt)

    def test_sha256_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.bin"
            path.write_bytes(b"abc")
            self.assertEqual(
                sha256_file(path),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            )


if __name__ == "__main__":
    unittest.main()
