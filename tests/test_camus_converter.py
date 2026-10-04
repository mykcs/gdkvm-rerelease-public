import json
import tempfile
import unittest
from pathlib import Path
import zipfile

import nibabel as nib
import numpy as np

from gdkvm_data.camus import (
    build_npy_cache_auto,
    convert_patient_archive,
    convert_patient_directory,
)


class CamusConverterTests(unittest.TestCase):
    def _write_nifti(self, path: Path, array: np.ndarray) -> None:
        affine = np.diag([0.4, 0.5, 1.0, 1.0])
        image = nib.Nifti1Image(array.astype(np.float32), affine=affine)
        image.header.set_zooms((0.4, 0.5, 1.0))
        nib.save(image, str(path))

    def _make_patient(self, root: Path, patient: str) -> Path:
        source = root / patient
        source.mkdir(parents=True)
        for view, frames in (("2CH", 18), ("4CH", 20)):
            image = np.zeros((12, 8, frames), dtype=np.float32)
            mask = np.zeros_like(image)
            for index in range(frames):
                image[:, :, index] = index
                mask[2:6, 2:5, index] = (index % 3) + 1
            self._write_nifti(
                source / f"{patient}_{view}_half_sequence.nii.gz",
                image,
            )
            self._write_nifti(
                source / f"{patient}_{view}_half_sequence_gt.nii.gz",
                mask,
            )
        return source

    def test_patient_archive_and_directory_match(self):
        patient = "patient0001"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self._make_patient(root, patient)
            archive_path = root / f"{patient}.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for path in source.glob("*.nii.gz"):
                    archive.write(path, arcname=f"{patient}/{path.name}")

            direct = convert_patient_directory(
                source,
                patient_id=patient,
                size=16,
                frame_count=10,
            )
            archived = convert_patient_archive(
                archive_path,
                patient_id=patient,
                size=16,
                frame_count=10,
            )

            self.assertEqual(len(direct), 2)
            self.assertEqual(len(archived), 2)
            for direct_row, archived_row in zip(direct, archived):
                np.testing.assert_array_equal(direct_row[0], archived_row[0])
                np.testing.assert_array_equal(direct_row[1], archived_row[1])
                self.assertEqual(direct_row[2], archived_row[2])

            by_view = {
                metadata["view"]: (frames, masks, metadata)
                for frames, masks, metadata in direct
            }
            frames_2ch, masks_2ch, meta_2ch = by_view["2CH"]
            self.assertEqual(frames_2ch.shape, (10, 16, 16))
            self.assertEqual(masks_2ch.shape, (10, 16, 16))
            self.assertEqual(frames_2ch.dtype, np.uint8)
            self.assertEqual(masks_2ch.dtype, np.uint8)
            self.assertEqual(meta_2ch["sampled_frame_indices_zero_based"][0], 0)
            self.assertEqual(meta_2ch["sampled_frame_indices_zero_based"][-1], 17)
            self.assertAlmostEqual(
                meta_2ch["source_spacing_axis01"][0], 0.4, places=6
            )
            self.assertAlmostEqual(
                meta_2ch["source_spacing_axis01"][1], 0.5, places=6
            )
            self.assertTrue(
                set(np.unique(masks_2ch)).issubset({0, 1, 2, 3})
            )

    def test_canonical_layout_builds_without_repacking_patient_zips(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "CAMUS_public"
            split_root = root / "database_split"
            nifti_root = root / "database_nifti"
            split_root.mkdir(parents=True)
            nifti_root.mkdir()

            all_patients = [f"patient{i:04d}" for i in range(1, 501)]
            split_root.joinpath("subgroup_training.txt").write_text(
                "\n".join(all_patients[:400]) + "\n"
            )
            split_root.joinpath("subgroup_validation.txt").write_text(
                "\n".join(all_patients[400:450]) + "\n"
            )
            split_root.joinpath("subgroup_testing.txt").write_text(
                "\n".join(all_patients[450:]) + "\n"
            )
            self._make_patient(nifti_root, "patient0001")

            output = Path(tmp) / "cache"
            build_npy_cache_auto(
                root,
                output,
                split="train",
                size=16,
                frame_count=10,
                patient_limit=1,
            )

            frames = np.load(output / "frames.npy", mmap_mode="r")
            masks = np.load(output / "masks.npy", mmap_mode="r")
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(frames.shape, (2, 10, 16, 16))
            self.assertEqual(masks.shape, (2, 10, 16, 16))
            self.assertEqual(
                manifest["source_layout"],
                "canonical-nifti-directory",
            )
            self.assertTrue(manifest["partial_source"])
            self.assertEqual(manifest["patient_count"], 1)


if __name__ == "__main__":
    unittest.main()
