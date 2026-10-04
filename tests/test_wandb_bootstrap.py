import unittest

from gdkvm_observability.bootstrap import build_wandb_init_kwargs, default_run_name
from gdkvm_observability.logger import RunIdentity


class WandbBootstrapTests(unittest.TestCase):
    def _identity(self):
        return RunIdentity(
            code_sha="abcdef123456",
            eval_protocol_id="gdkvm-rerelease-v1",
            dataset_release_id="camus-official",
            split_id="official",
            checkpoint_sha256="none",
            runtime_profile="eager",
            seed=7,
            world_size=1,
            compile_mode="eager",
        )

    def test_init_kwargs_bind_identity_and_deduplicate_tags(self):
        kwargs = build_wandb_init_kwargs(
            self._identity(),
            project="GDKVM-rerelease",
            entity="team",
            name="run-name",
            config={"model": "gdkvm"},
            tags=["rerelease-v1", "eager", "eager"],
        )
        self.assertEqual(kwargs["config"]["identity/code_sha"], "abcdef123456")
        self.assertEqual(kwargs["config"]["model"], "gdkvm")
        self.assertEqual(kwargs["tags"], ["rerelease-v1", "eager"])

    def test_identity_keys_cannot_be_overwritten(self):
        with self.assertRaises(ValueError):
            build_wandb_init_kwargs(
                self._identity(),
                project="x",
                config={"identity/code_sha": "wrong"},
            )

    def test_default_name_is_stable(self):
        self.assertEqual(
            default_run_name(
                dataset="camus",
                model="gdkvm",
                runtime="eager",
                seed=7,
                code_sha="abcdef123456",
            ),
            "camus-gdkvm-eager-s7-abcdef12",
        )


if __name__ == "__main__":
    unittest.main()
