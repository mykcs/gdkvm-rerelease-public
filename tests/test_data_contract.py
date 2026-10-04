import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from gdkvm_data.contract import (
    CAMUS_EXPECTED_PATIENTS,
    ECHONET_EXPECTED_VIDEOS,
    expand_camus_views,
    load_camus_split_dir,
    load_echonet_filelist,
    uniform_cyclic_frame_indices,
    uniform_frame_indices,
    validate_disjoint_splits,
)
from gdkvm_data.npy_cache import NpyCacheReader, write_npy_cache


class SamplingContractTests(unittest.TestCase):
    def test_uniform_sampling_includes_endpoints(self):
        indices = uniform_frame_indices(18, 10)
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 17)
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(np.unique(indices)), 10)
        self.assertTrue(np.all(np.diff(indices) > 0))

    def test_uniform_sampling_can_bind_ed_es_interval(self):
        indices = uniform_frame_indices(40, 10, start=7, stop=30)
        self.assertEqual(indices[0], 7)
        self.assertEqual(indices[-1], 30)
        self.assertTrue(np.all(indices >= 7))
        self.assertTrue(np.all(indices <= 30))

    def test_uniform_sampling_rejects_short_interval(self):
        with self.assertRaises(ValueError):
            uniform_frame_indices(8, 10)

    def test_cyclic_sampling_wraps_ed_to_es(self):
        indices = uniform_cyclic_frame_indices(30, 24, 8, 10)
        self.assertEqual(indices[0], 24)
        self.assertEqual(indices[-1], 8)
        self.assertEqual(len(indices), 10)
        self.assertEqual(len(np.unique(indices)), 10)
        # Every step moves forward around the circular video timeline.
        deltas = (np.diff(indices) % 30)
        self.assertTrue(np.all(deltas > 0))

    def test_cyclic_sampling_allows_repeat_for_short_systolic_interval(self):
        indices = uniform_cyclic_frame_indices(109, 28, 36, 10)
        self.assertEqual(indices[0], 28)
        self.assertEqual(indices[-1], 36)
        self.assertEqual(len(indices), 10)
        self.assertLess(len(np.unique(indices)), 10)
        self.assertTrue(np.all(indices >= 28))
        self.assertTrue(np.all(indices <= 36))


class SplitContractTests(unittest.TestCase):
    def test_disjoint_split_rejects_leakage(self):
        with self.assertRaises(ValueError):
            validate_disjoint_splits({"train": ["p1"], "val": ["p1"], "test": []})

    def test_camus_official_counts_and_view_expansion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            groups = {
                "training": [f"patient{i:04d}" for i in range(1, 401)],
                "validation": [f"patient{i:04d}" for i in range(401, 451)],
                "testing": [f"patient{i:04d}" for i in range(451, 501)],
            }
            for suffix, values in groups.items():
                (root / f"subgroup_{suffix}.txt").write_text(
                    "\n".join(values) + "\n", encoding="utf-8"
                )
            splits = load_camus_split_dir(root)
            self.assertEqual(
                {key: len(value) for key, value in splits.items()},
                CAMUS_EXPECTED_PATIENTS,
            )
            views = expand_camus_views(splits)
            self.assertEqual(len(views["train"]), 800)
            self.assertEqual(len(views["val"]), 100)
            self.assertEqual(len(views["test"]), 100)
            self.assertEqual(views["train"][0], "patient0001_2CH")
            self.assertEqual(views["train"][1], "patient0001_4CH")

    def test_echonet_filelist_uses_filelist_split_authority(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "FileList.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=["FileName", "Split"])
                writer.writeheader()
                for split, count in ECHONET_EXPECTED_VIDEOS.items():
                    for i in range(count):
                        writer.writerow({"FileName": f"{split.lower()}-{i}", "Split": split})
            groups = load_echonet_filelist(path)
            self.assertEqual(
                {key: len(value) for key, value in groups.items()},
                ECHONET_EXPECTED_VIDEOS,
            )


class NpyCacheTests(unittest.TestCase):
    def test_cache_roundtrip_and_mmap(self):
        rng = np.random.default_rng(7)
        frames = rng.integers(0, 256, size=(4, 10, 16, 16), dtype=np.uint8)
        masks = (frames > 127).astype(np.uint8)
        metadata = [{"sample_id": f"s{i}"} for i in range(4)]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            write_npy_cache(
                root,
                frames,
                masks,
                metadata,
                manifest_extra={"protocol_id": "fixture"},
            )
            cache = NpyCacheReader(root, mmap=True)
            self.assertEqual(len(cache), 4)
            np.testing.assert_array_equal(cache[2][0], frames[2])
            np.testing.assert_array_equal(cache[2][1], masks[2])
            self.assertEqual(cache[2][2]["sample_id"], "s2")
            self.assertEqual(cache.manifest["protocol_id"], "fixture")
            multiclass = cache.masks.copy()
            multiclass[2, 0, :2, :2] = 2
            np.save(root / "masks.npy", multiclass, allow_pickle=False)
            cache = NpyCacheReader(root, mmap=False)
            binary = cache.binary_masks(2, label=1)
            self.assertEqual(binary.dtype, np.uint8)
            self.assertTrue(set(np.unique(binary)).issubset({0, 1}))
            self.assertEqual(int(binary[0, 0, 0]), 0)


if __name__ == "__main__":
    unittest.main()
