import math
import unittest

import numpy as np

from gdkvm_eval.metrics import (
    PROTOCOL_ID,
    aggregate_segmentation,
    bland_altman_summary,
    evaluate_segmentation_case,
    lvef_summary,
    patient_bootstrap_mean_ci,
    regression_summary,
)


class SegmentationMetricsTests(unittest.TestCase):
    def test_protocol_id(self):
        self.assertEqual(PROTOCOL_ID, "gdkvm-rerelease-v1")

    def test_identical_mask(self):
        ref = np.zeros((16, 16), dtype=np.uint8)
        ref[4:12, 5:11] = 1
        result = evaluate_segmentation_case(ref, ref)
        self.assertEqual(result.status, "ok")
        self.assertAlmostEqual(result.dice, 1.0)
        self.assertAlmostEqual(result.iou, 1.0)
        self.assertAlmostEqual(result.hd, 0.0)
        self.assertAlmostEqual(result.hd95, 0.0)
        self.assertAlmostEqual(result.asd, 0.0)

    def test_single_pixel_shift(self):
        ref = np.zeros((8, 8), dtype=np.uint8)
        pred = np.zeros_like(ref)
        ref[3, 3] = 1
        pred[3, 4] = 1
        result = evaluate_segmentation_case(pred, ref)
        self.assertAlmostEqual(result.dice, 0.0)
        self.assertAlmostEqual(result.iou, 0.0)
        self.assertAlmostEqual(result.hd, 1.0)
        self.assertAlmostEqual(result.hd95, 1.0)
        self.assertAlmostEqual(result.asd, 1.0)

    def test_anisotropic_spacing(self):
        ref = np.zeros((8, 8), dtype=np.uint8)
        pred = np.zeros_like(ref)
        ref[2, 3] = 1
        pred[3, 3] = 1
        result = evaluate_segmentation_case(
            pred, ref, spacing=(2.0, 0.5), distance_unit="mm"
        )
        self.assertEqual(result.distance_unit, "mm")
        self.assertAlmostEqual(result.hd, 2.0)
        self.assertAlmostEqual(result.hd95, 2.0)
        self.assertAlmostEqual(result.asd, 2.0)

    def test_empty_mask_policy(self):
        ref = np.zeros((10, 20), dtype=np.uint8)
        ref[3:7, 5:12] = 1
        empty = np.zeros_like(ref)

        missed = evaluate_segmentation_case(empty, ref)
        expected = math.sqrt(9**2 + 19**2)
        self.assertEqual(missed.status, "one_empty")
        self.assertAlmostEqual(missed.dice, 0.0)
        self.assertAlmostEqual(missed.hd95, expected)

        absent = evaluate_segmentation_case(empty, empty)
        self.assertEqual(absent.status, "both_empty")
        self.assertTrue(math.isnan(absent.dice))
        self.assertTrue(math.isnan(absent.hd95))

    def test_aggregation_counts_and_is_partition_independent(self):
        ref = np.zeros((8, 8), dtype=np.uint8)
        ref[2:6, 2:6] = 1
        rows = [
            evaluate_segmentation_case(ref, ref),
            evaluate_segmentation_case(np.zeros_like(ref), ref),
            evaluate_segmentation_case(np.zeros_like(ref), np.zeros_like(ref)),
        ]
        whole = aggregate_segmentation(rows)
        reconstructed = aggregate_segmentation(rows[:1] + rows[1:])
        self.assertEqual(whole, reconstructed)
        self.assertEqual(whole["n_total"], 3)
        self.assertEqual(whole["n_valid"], 2)
        self.assertEqual(whole["n_one_empty"], 1)
        self.assertEqual(whole["n_both_empty"], 1)
        self.assertAlmostEqual(whole["dice_mean"], 0.5)


class ClinicalMetricsTests(unittest.TestCase):
    def test_regression_summary(self):
        result = regression_summary([1.0, 2.0, 3.0], [1.0, 2.0, 4.0])
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["mae"], 1.0 / 3.0)
        self.assertAlmostEqual(result["rmse"], math.sqrt(1.0 / 3.0))
        self.assertAlmostEqual(result["r2"], 11.0 / 14.0)

    def test_bland_altman_direction_and_sample_sd(self):
        pred = [60.0, 50.0, 70.0]
        ref = [55.0, 52.0, 68.0]
        result = bland_altman_summary(pred, ref)
        expected_sd = float(np.std([5.0, -2.0, 2.0], ddof=1))
        self.assertEqual(result["difference"], "prediction-reference")
        self.assertAlmostEqual(result["bias"], 5.0 / 3.0)
        self.assertAlmostEqual(result["sample_sd"], expected_sd)
        self.assertAlmostEqual(
            result["loa_low"], 5.0 / 3.0 - 1.96 * expected_sd
        )
        self.assertAlmostEqual(
            result["loa_high"], 5.0 / 3.0 + 1.96 * expected_sd
        )

    def test_lvef_summary_uses_percentage_point_errors(self):
        result = lvef_summary([60.0, 50.0], [55.0, 52.0])
        self.assertEqual(result.n, 2)
        self.assertAlmostEqual(result.mae, 3.5)
        self.assertAlmostEqual(result.bias, 1.5)
        self.assertAlmostEqual(result.sample_sd, np.std([5.0, -2.0], ddof=1))

    def test_patient_bootstrap_weights_patients_not_rows(self):
        result = patient_bootstrap_mean_ci(
            [1.0, 3.0, 10.0],
            ["p1", "p1", "p2"],
            n_resamples=500,
            seed=7,
        )
        # p1 contributes mean 2; p2 contributes 10, so patient-macro mean is 6.
        self.assertEqual(result["n_patients"], 2)
        self.assertAlmostEqual(result["mean"], 6.0)
        self.assertLessEqual(result["ci_low"], result["mean"])
        self.assertGreaterEqual(result["ci_high"], result["mean"])


if __name__ == "__main__":
    unittest.main()
