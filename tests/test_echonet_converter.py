import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from gdkvm_data.echonet import convert_echonet_video, polygon_from_trace


class EchoNetConverterTests(unittest.TestCase):
    def _write_video(self, path: Path, frame_count: int = 20) -> None:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"MJPG"),
            20.0,
            (64, 64),
        )
        if not writer.isOpened():
            self.skipTest("OpenCV MJPG video writer unavailable")
        try:
            for index in range(frame_count):
                frame = np.full((64, 64, 3), index * 8, dtype=np.uint8)
                writer.write(frame)
        finally:
            writer.release()

    def test_polygon_from_trace_is_binary(self):
        rows = [
            (24.0, 12.0, 40.0, 12.0),
            (16.0, 28.0, 48.0, 28.0),
            (20.0, 48.0, 44.0, 48.0),
        ]
        mask = polygon_from_trace(rows, (64, 64))
        self.assertEqual(mask.dtype, np.uint8)
        self.assertGreater(int(mask.sum()), 0)
        self.assertTrue(set(np.unique(mask)).issubset({0, 1}))

    def test_convert_video_places_only_ed_es_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "fixture.avi"
            self._write_video(video)
            # Larger contour at frame 3 -> ED; smaller contour at 15 -> ES.
            traces = {
                3: [
                    (24.0, 10.0, 40.0, 10.0),
                    (12.0, 28.0, 52.0, 28.0),
                    (16.0, 52.0, 48.0, 52.0),
                ],
                15: [
                    (28.0, 20.0, 36.0, 20.0),
                    (22.0, 30.0, 42.0, 30.0),
                    (26.0, 42.0, 38.0, 42.0),
                ],
            }
            frames, masks, metadata = convert_echonet_video(
                video, traces, size=32, frame_count=10
            )
            self.assertEqual(frames.shape, (10, 32, 32))
            self.assertEqual(masks.shape, (10, 32, 32))
            self.assertEqual(frames.dtype, np.uint8)
            self.assertEqual(masks.dtype, np.uint8)
            self.assertEqual(metadata["ed_index"], 3)
            self.assertEqual(metadata["es_index"], 15)
            self.assertEqual(metadata["sampled_indices"][0], 3)
            self.assertEqual(metadata["sampled_indices"][-1], 15)
            self.assertGreater(int(masks[0].sum()), int(masks[-1].sum()))
            self.assertEqual(int(masks[1:-1].sum()), 0)


if __name__ == "__main__":
    unittest.main()
