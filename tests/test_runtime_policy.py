import json
from pathlib import Path
import unittest

from gdkvm_runtime.policy import (
    COMPILE_MODES,
    RuntimePolicy,
    prepare_model_runtime,
    resolve_runtime_policy,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeModel:
    def __init__(self):
        self.memory_formats = []

    def to(self, *, memory_format=None):
        self.memory_formats.append(memory_format)
        return self


class FakeCudnn:
    benchmark = None


class FakeBackends:
    cudnn = FakeCudnn()


class FakeDynamoConfig:
    optimize_ddp = None


class FakeDynamo:
    config = FakeDynamoConfig()


class FakeTorch:
    channels_last = "channels_last"
    backends = FakeBackends()
    _dynamo = FakeDynamo()

    def __init__(self):
        self.compile_calls = []

    def compile(self, model, **kwargs):
        self.compile_calls.append(kwargs)
        return ("compiled", model, kwargs)


class RuntimePolicyTests(unittest.TestCase):
    def test_default_is_eager_reference(self):
        policy = resolve_runtime_policy({})
        self.assertEqual(policy.profile_id, "eager-reference")
        self.assertFalse(policy.compile_enabled)
        self.assertFalse(policy.channels_last)
        self.assertFalse(policy.cudnn_benchmark)

    def test_unknown_compile_mode_fails_closed(self):
        with self.assertRaises(ValueError):
            resolve_runtime_policy(
                {"runtime": {"compile": {"enabled": True, "mode": "magic"}}}
            )

    def test_eager_path_never_calls_compile(self):
        torch = FakeTorch()
        model = FakeModel()
        policy = RuntimePolicy(compile_enabled=False, channels_last=True)
        out = prepare_model_runtime(model, torch, policy)
        self.assertIs(out, model)
        self.assertEqual(torch.compile_calls, [])
        self.assertEqual(model.memory_formats, ["channels_last"])

    def test_compile_is_explicit_and_parameterized(self):
        torch = FakeTorch()
        model = FakeModel()
        policy = RuntimePolicy(
            profile_id="fixture-compiled",
            compile_enabled=True,
            compile_mode="reduce-overhead",
            compile_fullgraph=True,
            compile_dynamic=True,
            channels_last=False,
            cudnn_benchmark=True,
            optimize_ddp=False,
        )
        out = prepare_model_runtime(model, torch, policy)
        self.assertEqual(out[0], "compiled")
        self.assertEqual(
            torch.compile_calls,
            [
                {
                    "mode": "reduce-overhead",
                    "fullgraph": True,
                    "dynamic": True,
                }
            ],
        )
        self.assertTrue(torch.backends.cudnn.benchmark)
        self.assertFalse(torch._dynamo.config.optimize_ddp)

    def test_profile_registry_uses_supported_compile_modes(self):
        registry = json.loads((ROOT / "runtime" / "profiles.json").read_text())
        self.assertTrue(registry["rules"]["eager_is_reference"])
        declared = set(registry["compile_modes"])
        self.assertIn("eager", declared)
        self.assertTrue((declared - {"eager"}).issubset(COMPILE_MODES))

    def test_historical_lane_contains_captured_2_6_cu118(self):
        registry = json.loads((ROOT / "runtime" / "profiles.json").read_text())
        candidates = registry["lanes"]["historical"]["candidates"]
        self.assertTrue(
            any(
                row["pytorch"] == "2.6.0"
                and row["cuda_wheel"] == "cu118"
                and row["status"] == "historical-captured-environment"
                for row in candidates
            )
        )

    def test_modern_lane_tracks_current_2_14_cu130_candidate(self):
        registry = json.loads((ROOT / "runtime" / "profiles.json").read_text())
        candidates = registry["lanes"]["modern"]["candidates"]
        self.assertTrue(
            any(
                row["pytorch"] == "2.14.0"
                and row["cuda_wheel"] == "cu130"
                for row in candidates
            )
        )

    def test_trainer_no_longer_forces_compile(self):
        trainer = (ROOT / "model" / "trainer.py").read_text(encoding="utf-8")
        self.assertIn("resolve_runtime_policy", trainer)
        self.assertIn("prepare_model_runtime", trainer)
        self.assertNotIn("torch.compile(model", trainer)
        self.assertNotIn("torch._dynamo.config.optimize_ddp", trainer)

    def test_model_channels_last_is_policy_controlled(self):
        model = (ROOT / "model" / "gdkvm01.py").read_text(encoding="utf-8")
        self.assertIn("use_channels_last: bool=False", model)
        self.assertIn("self.use_channels_last", model)
        self.assertIn(
            "if self.use_channels_last and images_DCHW.device.type == 'cuda':",
            model,
        )


if __name__ == "__main__":
    unittest.main()
