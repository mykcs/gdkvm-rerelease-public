import math
import unittest

import numpy as np
from skimage.measure import regionprops

from gdkvm_eval import (
    CLINICAL_METHOD_ID,
    area_length_lvef_percent,
    area_length_volume,
)


class ClinicalEvaluationTests(unittest.TestCase):
    def test_method_id_is_versioned(self):
        self.assertEqual(CLINICAL_METHOD_ID, "single-view-area-length-v1")

    def test_area_length_volume_matches_declared_formula(self):
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[3:13, 5:11] = 1
        region = regionprops(mask)[0]
        expected = (8.0 * float(region.area) ** 2) / (
            3.0 * math.pi * float(region.major_axis_length)
        )
        self.assertAlmostEqual(area_length_volume(mask), expected, places=12)

    def test_lvef_is_reported_in_percent(self):
        ed = np.zeros((24, 24), dtype=np.uint8)
        es = np.zeros((24, 24), dtype=np.uint8)
        ed[4:20, 6:18] = 1
        es[7:17, 8:16] = 1
        edv = area_length_volume(ed)
        esv = area_length_volume(es)
        expected = 100.0 * (edv - esv) / edv
        self.assertAlmostEqual(
            area_length_lvef_percent(ed, es), expected, places=12
        )

    def test_empty_ed_is_not_a_fake_zero_lvef(self):
        empty = np.zeros((8, 8), dtype=np.uint8)
        self.assertTrue(math.isnan(area_length_lvef_percent(empty, empty)))

    def test_non_2d_mask_fails_closed(self):
        with self.assertRaises(ValueError):
            area_length_volume(np.zeros((2, 3, 4), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
