from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class TrainerDependencyBoundaryTests(unittest.TestCase):
    def test_visualization_is_not_a_top_level_trainer_dependency(self):
        text = (ROOT / "model" / "trainer.py").read_text(encoding="utf-8")
        import_line = "from vis.vis_0730 import visualize_sequence"
        self.assertEqual(text.count(import_line), 1)

        top_level = text.split("class Trainer", 1)[0]
        self.assertNotIn(import_line, top_level)

        visualize_method = text.split("def _visualize_batch", 1)[1]
        self.assertIn(import_line, visualize_method)

    def test_trainer_has_no_direct_wandb_dependency(self):
        text = (ROOT / "model" / "trainer.py").read_text(encoding="utf-8")
        self.assertNotIn("import wandb", text)
        self.assertNotIn("wandb.", text)
        self.assertIn("ExperimentLogger", text)

    def test_tensorboard_and_pillow_are_optional_backend_dependencies(self):
        trainer = (ROOT / "model" / "trainer.py").read_text(encoding="utf-8")
        self.assertNotIn("TensorboardLogger", trainer)

        logger = (ROOT / "utils" / "logger.py").read_text(encoding="utf-8")
        top_level = logger.split("class TensorboardLogger", 1)[0]
        self.assertNotIn("torch.utils.tensorboard", top_level)
        self.assertNotIn("from PIL import Image", top_level)
        self.assertIn("from torch.utils.tensorboard import SummaryWriter", logger)
        self.assertIn("from PIL import Image", logger)

    def test_compile_policy_is_owned_by_runtime_layer_not_trainer(self):
        text = (ROOT / "model" / "trainer.py").read_text(encoding="utf-8")
        policy = (ROOT / "gdkvm_runtime" / "policy.py").read_text(encoding="utf-8")

        self.assertNotIn("torch.compile(", text)
        self.assertNotIn("torch._dynamo.config.optimize_ddp", text)
        self.assertIn("resolve_runtime_policy", text)
        self.assertIn("prepare_model_runtime", text)

        self.assertIn("compile_enabled", policy)
        self.assertIn("compile_mode", policy)
        self.assertIn("reduce-overhead", policy)
        self.assertIn("max-autotune", policy)


if __name__ == "__main__":
    unittest.main()
