"""Tests for webcam capture, ground-truth labeling, and manifest management."""

import csv
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from pharmacy_assistant.capture import (
    MANIFEST_HEADER,
    capture_and_label_interactive,
    next_sample_basename,
    save_labeled_sample,
)
from pharmacy_assistant.prescription import CameraCaptureCancelled


class CaptureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.training_root = Path(self.temp_dir.name)
        (self.training_root / "manifest.csv").touch()
        self.dummy_image = np.full((100, 100, 3), 200, dtype=np.uint8)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_next_sample_basename_increments_properly(self) -> None:
        eval_dir = self.training_root / "evaluation"
        self.assertEqual(next_sample_basename(eval_dir), "sample001")
        (eval_dir / "sample001.png").touch()
        self.assertEqual(next_sample_basename(eval_dir), "sample002")
        (eval_dir / "sample005.png").touch()
        self.assertEqual(next_sample_basename(eval_dir), "sample006")

    def test_save_labeled_sample_creates_png_gt_and_manifest_entry(self) -> None:
        img_path, gt_path, source_id = save_labeled_sample(
            image=self.dummy_image,
            ground_truth_text="PARACETAMOL 500 mg",
            target_class="PARACETAMOL",
            conditions="overhead light, flat",
            split="test",
            source_photo_id="photo_p1",
            license_origin="operator_capture",
            training_root=self.training_root,
        )

        self.assertTrue(img_path.is_file())
        self.assertTrue(gt_path.is_file())
        self.assertEqual(source_id, "photo_p1")
        self.assertEqual(gt_path.read_text(encoding="utf-8"), "PARACETAMOL 500 mg\n")

        # Check evaluation directory location for test split
        self.assertEqual(img_path.parent.name, "evaluation")

        # Verify manifest row
        manifest_path = self.training_root / "manifest.csv"
        with manifest_path.open(encoding="utf-8") as f:
            reader = list(csv.reader(f))

        self.assertEqual(reader[0], list(MANIFEST_HEADER))
        self.assertEqual(
            reader[1],
            [
                "photo_p1",
                f"evaluation/{img_path.name}",
                f"evaluation/{gt_path.name}",
                "test",
                "PARACETAMOL",
                "overhead light, flat",
                "operator_capture",
            ],
        )

    def test_save_labeled_sample_rejects_empty_ground_truth(self) -> None:
        with self.assertRaises(ValueError):
            save_labeled_sample(
                image=self.dummy_image,
                ground_truth_text="   ",
                target_class="PARACETAMOL",
                conditions="",
                split="test",
                source_photo_id=None,
                license_origin="",
                training_root=self.training_root,
            )

    def test_interactive_capture_with_mock_inputs(self) -> None:
        # Simulate operator typing:
        # 1. Exact text: "AMOXICILLIN 250 mg capsule"
        # 2. Class: [Enter for default]
        # 3. Conditions: "dim light"
        # 4. Source ID: "src_amox_01"
        # 5. Origin: [Enter for default]
        simulated_input = io.StringIO(
            "AMOXICILLIN 250 mg capsule\n\ndim light\nsrc_amox_01\n\n"
        )

        with patch(
            "pharmacy_assistant.capture.capture_webcam_image",
            return_value=self.dummy_image,
        ):
            result = capture_and_label_interactive(
                camera_index=0,
                split="test",
                training_root=self.training_root,
                input_stream=simulated_input,
            )

        self.assertIsNotNone(result)
        img_path, gt_path, source_id = result
        self.assertTrue(img_path.is_file())
        self.assertEqual(source_id, "src_amox_01")
        self.assertEqual(
            gt_path.read_text(encoding="utf-8"),
            "AMOXICILLIN 250 mg capsule\n",
        )

    def test_interactive_capture_cancelled(self) -> None:
        with patch(
            "pharmacy_assistant.capture.capture_webcam_image",
            side_effect=CameraCaptureCancelled("Cancelled by user"),
        ):
            result = capture_and_label_interactive(
                camera_index=0,
                split="test",
                training_root=self.training_root,
            )
            self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
