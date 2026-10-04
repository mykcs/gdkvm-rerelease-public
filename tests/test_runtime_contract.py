import ast
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class RuntimeContractTests(unittest.TestCase):
    def test_profiles_have_three_roles_and_eager_reference(self):
        data = json.loads((ROOT / "runtime" / "profiles.json").read_text())
        self.assertEqual(set(data["lanes"]), {"historical", "compatibility", "modern"})
        self.assertIn("eager", data["compile_modes"])
        self.assertTrue(data["rules"]["eager_is_reference"])
        self.assertFalse(data["rules"]["host_driver_change_authorized"])
        self.assertTrue(data["rules"]["compatibility_3090_requires_real_3090_qualification"])

    def test_compatibility_and_blackwell_lanes_are_intentionally_separate(self):
        data = json.loads((ROOT / "runtime" / "profiles.json").read_text())
        compat = data["lanes"]["compatibility"]["candidates"][0]
        modern = data["lanes"]["modern"]["candidates"][0]
        self.assertEqual(compat["cuda_wheel"], "cu126")
        self.assertIn(modern["cuda_wheel"], {"cu130", "cu132"})
        self.assertNotEqual(compat["pytorch"], modern["pytorch"])
        self.assertTrue(data["rules"]["compatibility_3090_requires_real_3090_qualification"])

    def test_runtime_benchmark_is_synthetic_and_does_not_modify_host(self):
        path = ROOT / "scripts" / "benchmark_gdkvm_runtime.py"
        text = path.read_text()
        tree = ast.parse(text)
        self.assertIn("synthetic data", text.lower())
        forbidden = ("apt ", "apt-get", "nvidia-smi -pm", "modprobe", "docker ")
        for needle in forbidden:
            self.assertNotIn(needle, text)
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        self.assertNotIn("wandb", imports)

    def test_model_metadata_is_normalized_before_compile_boundary(self):
        trainer = (ROOT / "model" / "trainer.py").read_text()
        model = (ROOT / "model" / "gdkvm01.py").read_text()
        benchmark = (ROOT / "scripts" / "benchmark_gdkvm_runtime.py").read_text()
        self.assertIn('info["num_objects"] = [', trainer)
        self.assertNotIn("num.item() for num in data['info']['num_objects']", model)
        self.assertIn("must be normalized to Python integers", model)
        self.assertIn('"info": {"num_objects": [1]}', benchmark)

    def test_attention_mask_avoids_dynamic_nonzero_indexing(self):
        text = (ROOT / "model" / "transformer" / "object_transformer.py").read_text()
        self.assertNotIn("torch.where(aux_mask.sum", text)
        self.assertIn("fully_blocked = aux_mask.all(dim=-1, keepdim=True)", text)
        self.assertIn("aux_mask = aux_mask & ~fully_blocked", text)

    def test_compile_modes_are_explicit_cli_inputs(self):
        text = (ROOT / "scripts" / "benchmark_gdkvm_runtime.py").read_text()
        self.assertIn('"--modes"', text)
        self.assertIn('torch.compile(model, mode=mode)', text)
        self.assertIn('"comparison_to_eager"', text)
        self.assertIn('"cold_start_s"', text)
        self.assertIn('"peak_memory_allocated_bytes"', text)


if __name__ == "__main__":
    unittest.main()
