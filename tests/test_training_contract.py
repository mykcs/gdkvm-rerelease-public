import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gdkvm_training.contract import (
    cache_manifest_identity,
    checkpoint_identity,
    json_safe,
    memory_fraction_for_limit,
    validate_formal_cache_identity,
)


ROOT = Path(__file__).resolve().parents[1]


class TrainingContractTests(unittest.TestCase):
    def test_23gib_limit_on_32gib_gpu_is_bounded(self):
        fraction = memory_fraction_for_limit(23.0, 32 * 1024**3)
        self.assertAlmostEqual(fraction, 23 / 32)
        self.assertLess(fraction, 1.0)

    def test_memory_limit_never_exceeds_full_allocator(self):
        self.assertEqual(
            memory_fraction_for_limit(23.0, 16 * 1024**3),
            1.0,
        )

    def test_checkpoint_identity_hashes_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "new.pth"
            path.write_bytes(b"new-gdkvm-checkpoint")
            row = checkpoint_identity(path, role="engineering-smoke")
            self.assertEqual(row["byte_size"], len(b"new-gdkvm-checkpoint"))
            self.assertEqual(len(row["sha256"]), 64)
            self.assertEqual(row["role"], "engineering-smoke")

    def test_json_safe_normalizes_numpy_values_and_nonfinite_numbers(self):
        payload = {
            "float32": np.float32(0.9367),
            "int64": np.int64(7),
            "array": np.array([np.float32(1.0), np.float32(2.0)]),
            "nan": float("nan"),
            "inf32": np.float32(np.inf),
        }
        safe = json_safe(payload)
        self.assertIsInstance(safe["float32"], float)
        self.assertEqual(safe["int64"], 7)
        self.assertEqual(safe["array"], [1.0, 2.0])
        self.assertIsNone(safe["nan"])
        self.assertIsNone(safe["inf32"])
        json.dumps(safe, allow_nan=False)

    def test_cache_identity_uses_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "dataset": "CAMUS",
                        "split": "train",
                        "format": "contiguous-npy",
                        "shape": [8, 10, 64, 64],
                    }
                )
            )
            row = cache_manifest_identity(root)
            self.assertEqual(row["dataset"], "CAMUS")
            self.assertEqual(row["split"], "train")
            self.assertEqual(row["sample_count"], 8)
            self.assertEqual(len(row["manifest_sha256"]), 64)

    def test_formal_cache_rejects_partial_source(self):
        identity = {
            "dataset": "CAMUS",
            "split": "train",
            "sample_count": 800,
            "partial_source": True,
            "selection": None,
            "source_is_full_video_tree": None,
        }
        with self.assertRaises(ValueError):
            validate_formal_cache_identity(
                identity,
                role="train",
                expected_dataset="CAMUS",
                expected_samples=800,
            )

    def test_formal_cache_rejects_wrong_count_or_split(self):
        base = {
            "dataset": "CAMUS",
            "split": "train",
            "sample_count": 800,
            "partial_source": False,
            "selection": None,
            "source_is_full_video_tree": None,
        }
        validate_formal_cache_identity(
            base,
            role="train",
            expected_dataset="CAMUS",
            expected_samples=800,
        )
        with self.assertRaises(ValueError):
            validate_formal_cache_identity(
                {**base, "sample_count": 799},
                role="train",
                expected_dataset="CAMUS",
                expected_samples=800,
            )
        with self.assertRaises(ValueError):
            validate_formal_cache_identity(
                {**base, "split": "val"},
                role="train",
                expected_dataset="CAMUS",
                expected_samples=800,
            )

    def test_formal_echonet_cache_requires_full_source_tree(self):
        identity = {
            "dataset": "EchoNet-Dynamic",
            "split": "TRAIN",
            "sample_count": 7460,
            "partial_source": False,
            "selection": {"split": "TRAIN", "limit_samples": None},
            "source_is_full_video_tree": False,
        }
        with self.assertRaises(ValueError):
            validate_formal_cache_identity(
                identity,
                role="train",
                expected_dataset="EchoNet-Dynamic",
                expected_samples=7460,
            )

    def test_reference_configs_are_retrain_not_historical_replay(self):
        for name in ("camus-v1.yaml", "echonet-v1.yaml"):
            text = (ROOT / "training" / "configs" / name).read_text()
            self.assertIn("profile_id: 3090-like-eager-fp32", text)
            self.assertIn("memory_limit_gib: 23.0", text)
            self.assertIn("batch_size: 10", text)
            self.assertIn("expected_samples:", text)
            self.assertIn("enabled: false", text)
            self.assertIn("amp: false", text)
            self.assertIn("seq_length: 10", text)
            self.assertIn("pretrained_backbones: true", text)

    def test_training_requirements_are_self_contained(self):
        requirements = (ROOT / "requirements-train.txt").read_text().splitlines()
        includes = [
            line.split(maxsplit=1)[1]
            for line in requirements
            if line.strip().startswith("-r ")
        ]
        self.assertEqual(includes, [])
        self.assertIn(
            "scipy>=1.16.3,<2",
            requirements,
        )

    def test_trainer_has_non_ddp_state_path(self):
        text = (ROOT / "model" / "trainer.py").read_text()
        self.assertIn("if self.is_distributed:", text)
        self.assertIn("self.model = model", text)
        self.assertIn("def _model_for_state", text)
        self.assertNotIn('"model": self.model.module.state_dict()', text)


    def test_integrator_is_safe_without_initialized_process_group(self):
        text = (ROOT / "utils" / "log_integrator.py").read_text()
        self.assertIn("torch.distributed.is_initialized()", text)
        self.assertIn("else 0", text)
        self.assertIn("else 1", text)

    def test_training_entrypoint_emits_new_checkpoint_receipt(self):
        text = (ROOT / "scripts" / "train_gdkvm.py").read_text()
        self.assertIn('"kind": "gdkvm-rerelease-training"', text)
        self.assertIn("checkpoint_identity(", text)
        self.assertIn("memory_fraction_for_limit(", text)
        self.assertIn("validate_formal_cache_identity(", text)
        self.assertIn("formal_cache_identities", text)
        self.assertIn("allow_nan=False", text)
        self.assertNotIn("historical checkpoint", text.lower())

    def test_checkpoint_evaluation_entrypoint_is_protocol_bound(self):
        text = (ROOT / "scripts" / "evaluate_gdkvm_checkpoint.py").read_text()
        self.assertIn("PROTOCOL_ID", text)
        self.assertIn("validate_formal_cache_identity(", text)
        self.assertIn("checkpoint_identity(", text)
        self.assertIn("patient_bootstrap_mean_ci(", text)
        self.assertIn("allow_nan=False", text)
        self.assertIn("pretrained_backbones=False", text)


if __name__ == "__main__":
    unittest.main()
