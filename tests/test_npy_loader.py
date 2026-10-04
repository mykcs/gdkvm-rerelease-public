import tempfile
import unittest
from pathlib import Path

import numpy as np

from gdkvm_data.loader import GDKVMNpyDataset
from gdkvm_data.npy_cache import write_npy_cache


class GDKVMNpyDatasetTests(unittest.TestCase):
    def test_model_facing_shapes_and_label_projection(self):
        frames = np.arange(2 * 10 * 8 * 8, dtype=np.uint8).reshape(2, 10, 8, 8)
        masks = np.zeros_like(frames, dtype=np.uint8)
        masks[0, :, 2:5, 3:6] = 1
        masks[0, :, 0:2, 0:2] = 2
        metadata = [{"sample_id": "patient0001_2CH"}, {"sample_id": "empty"}]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            write_npy_cache(root, frames, masks, metadata)

            dataset = GDKVMNpyDataset(root, label=1)
            item = dataset[0]

            self.assertEqual(item["rgb"].shape, (10, 1, 8, 8))
            self.assertEqual(item["rgb"].dtype, np.float32)
            self.assertGreaterEqual(float(item["rgb"].min()), 0.0)
            self.assertLessEqual(float(item["rgb"].max()), 1.0)
            self.assertEqual(item["cls_gt"].shape, (10, 1, 8, 8))
            self.assertEqual(item["cls_gt"].dtype, np.int64)
            self.assertEqual(item["ff_gt"].shape, (1, 1, 8, 8))
            self.assertEqual(item["selector"].tolist(), [1.0])
            self.assertEqual(item["info"]["num_objects"], 1)
            self.assertEqual(item["info"]["name"], "patient0001_2CH")
            self.assertEqual(int(item["cls_gt"].max()), 1)
            self.assertEqual(int(item["cls_gt"][:, 0, 0:2, 0:2].sum()), 0)

            empty = dataset[1]
            self.assertEqual(empty["selector"].tolist(), [0.0])
            self.assertEqual(empty["info"]["num_objects"], 0)

    def test_uint8_mode_keeps_cache_values(self):
        frames = np.full((1, 10, 4, 4), 127, dtype=np.uint8)
        masks = np.zeros_like(frames)
        metadata = [{"sample_id": "sample"}]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "cache"
            write_npy_cache(root, frames, masks, metadata)
            item = GDKVMNpyDataset(root, normalize_images=False)[0]
            self.assertEqual(item["rgb"].dtype, np.uint8)
            self.assertEqual(int(item["rgb"][0, 0, 0, 0]), 127)


if __name__ == "__main__":
    unittest.main()
