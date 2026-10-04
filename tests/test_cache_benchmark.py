import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "benchmark_cache_formats", ROOT / "scripts" / "benchmark_cache_formats.py"
)
assert SPEC and SPEC.loader
bench = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench)


class TarCacheCompatibilityTests(unittest.TestCase):
    def test_tar_npy_roundtrip_is_lossless_and_readable(self):
        rng = np.random.default_rng(7)
        frames = rng.integers(0, 256, size=(4, 10, 8, 8), dtype=np.uint8)
        masks = (frames > 127).astype(np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "samples.tar"
            bench.save_tar(path, frames, masks)
            expected = bench.canonical_hash(frames, masks)
            self.assertEqual(bench.tar_hash(path), expected)
            value = bench.read_tar(path)
            self.assertIsInstance(value, int)


if __name__ == "__main__":
    unittest.main()
