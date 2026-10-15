"""Tests for conservative prescription image scanning and OCR."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytesseract

from pharmacy_assistant.cli import main
from pharmacy_assistant.database import connect, initialize_database, seed_demo_data
from pharmacy_assistant.matching import MedicineMatcher
from pharmacy_assistant.prescription import (
    CameraCaptureCancelled,
    CameraCaptureError,
    PrescriptionImageError,
    PrescriptionOCRError,
    TesseractUnavailableError,
    capture_webcam_image,
    load_valid_image,
    parse_prescription_text,
    scan_image,
    scan_image_array,
)
from pharmacy_assistant.repository import MedicineRepository


class PhaseTwoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_directory.name)
        self.image_path = self.directory / "prescription.png"
        image = np.full((500, 900, 3), 255, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.image_path), image))
        self.database_path = self.directory / "test.db"
        initialize_database(self.database_path)
        seed_demo_data(self.database_path)
        self.connection = connect(self.database_path)
        self.matcher = MedicineMatcher(self.connection)
        self.repository = MedicineRepository(self.connection)

    def tearDown(self) -> None:
        self.connection.close()
        self.temp_directory.cleanup()

    def test_accepts_supported_image_and_rejects_other_extensions(self) -> None:
        self.assertEqual(load_valid_image(self.image_path).shape, (500, 900, 3))
        for suffix in (".jpg", ".jpeg"):
            supported_image = self.directory / f"prescription{suffix}"
            pixels = np.full((40, 60, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(supported_image), pixels))
            self.assertEqual(load_valid_image(supported_image).shape, (40, 60, 3))
        unsupported = self.directory / "prescription.gif"
        unsupported.write_bytes(b"not an image")
        with self.assertRaises(PrescriptionImageError):
            load_valid_image(unsupported)

    def test_rejects_missing_and_unreadable_images(self) -> None:
        with self.assertRaises(PrescriptionImageError):
            load_valid_image(self.directory / "missing.jpg")
        unreadable = self.directory / "broken.jpeg"
        unreadable.write_bytes(b"not a JPEG image")
        with self.assertRaises(PrescriptionImageError):
            load_valid_image(unreadable)

    def test_parses_mocked_ocr_text_and_keeps_nonmedicine_lines_for_review(self) -> None:
        raw_text = (
            "Patient: Jane Doe\n"
            "1. Tab. Paracetamol 500 mg - 1-0-1\n"
            "Take after meals\n"
            "Cap Amoxicillin 250 mg"
        )
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            return_value=raw_text,
        ):
            result = scan_image(self.image_path, self.matcher, self.repository)

        self.assertEqual(result.raw_text, raw_text)
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].candidate.name, "Paracetamol")
        self.assertEqual(result.items[0].candidate.strength, "500 mg")
        self.assertEqual(result.items[0].database_status, "matched")
        self.assertEqual(result.items[0].location.code, "DEMO-A1")
        self.assertEqual(result.items[1].candidate.name, "Amoxicillin")
        self.assertEqual(result.items[1].candidate.strength, "250 mg")
        self.assertEqual(result.items[1].candidate.dosage_form, "capsule")
        self.assertEqual(
            result.unparsed_lines,
            ("Patient: Jane Doe", "Take after meals"),
        )

    def test_noisy_surrounding_punctuation_still_extracts_catalog_name(self) -> None:
        result = parse_prescription_text(
            r"FN Ba | | : if ! i 2 VCH ~5\ Par ACETAMOL | ~ -",
            self.matcher,
        )
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertEqual(candidate.name, "ParACETAMOL")
        self.assertIsNone(candidate.strength)
        self.assertIn("strength", candidate.review_fields)

    def test_noisy_name_keeps_identifiable_strength(self) -> None:
        result = parse_prescription_text(
            r"~5\ Par ACETAMOL | ~ - 500mg",
            self.matcher,
        )
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].name, "ParACETAMOL")
        self.assertEqual(result.candidates[0].strength, "500 mg")

    def test_ocr_tries_multiple_modes_and_selects_exact_catalog_evidence(self) -> None:
        outputs = [
            "FN Ba | | if ! VCH",
            "~5\\ Par ACETAMOL | ~ -",
            "PARACETAMOL",
            "PARACETAMOL",
            "PARACETAMOL",
            "PARACETAMOL",
        ]
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            side_effect=outputs,
        ) as ocr:
            result = scan_image_array(
                load_valid_image(self.image_path),
                self.matcher,
                self.repository,
            )

        self.assertEqual(ocr.call_count, 6)
        configurations = [call.kwargs["config"] for call in ocr.call_args_list]
        self.assertIn("--oem 3 --psm 6", configurations)
        self.assertIn("--oem 3 --psm 7", configurations)
        self.assertIn("--oem 3 --psm 11", configurations)
        self.assertIn("PARACETAMOL", result.raw_text)
        self.assertTrue(result.ocr_configuration)
        self.assertGreater(result.ocr_score, 0)

    def test_noisy_name_without_strength_stays_incomplete_for_review(self) -> None:
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            return_value=r"~5\ Par ACETAMOL | ~ -",
        ):
            result = scan_image_array(
                load_valid_image(self.image_path),
                self.matcher,
                self.repository,
            )

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].candidate.name, "ParACETAMOL")
        self.assertEqual(
            result.items[0].database_status,
            "ambiguous_requires_review",
        )
        self.assertIsNone(result.items[0].medicine)
        self.assertIsNone(result.items[0].location)
        self.assertIn("strength", result.items[0].review_fields)
        self.assertIn("medicine_match", result.items[0].review_fields)

    def test_debug_directory_saves_original_preprocessed_and_attempts(self) -> None:
        debug_directory = self.directory / "ocr-debug"
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            return_value="Paracetamol 500 mg",
        ):
            scan_image(
                self.image_path,
                self.matcher,
                self.repository,
                debug_dir=debug_directory,
            )

        self.assertTrue((debug_directory / "captured.png").is_file())
        self.assertTrue((debug_directory / "preprocessed.png").is_file())
        attempts = (debug_directory / "ocr-attempts.txt").read_text()
        self.assertIn("--psm 6", attempts)
        self.assertIn("--psm 7", attempts)
        self.assertIn("--psm 11", attempts)

    def test_ambiguous_database_match_requires_human_review(self) -> None:
        self.connection.execute(
            """
            INSERT INTO medicines (generic_name, brand_name, strength, dosage_form)
            VALUES ('paracetamol', 'Other Fictional Brand', '500 mg', 'capsule')
            """
        )
        self.connection.commit()
        raw_text = "Paracetamol 500 mg"
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            return_value=raw_text,
        ):
            result = scan_image(self.image_path, self.matcher, self.repository)

        self.assertEqual(result.items[0].database_status, "ambiguous")
        self.assertIsNone(result.items[0].medicine)
        self.assertIsNone(result.items[0].location)
        self.assertEqual(result.items[0].review_fields, ("medicine_match",))

    def test_scan_cli_displays_ocr_match_status_and_verified_location(self) -> None:
        raw_text = "Paracetamol tablet 500mg"
        output = io.StringIO()
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            return_value=raw_text,
        ), patch(
            "sys.argv",
            [
                "pharmacy-assistant",
                "scan",
                "--image",
                str(self.image_path),
                "--db",
                str(self.database_path),
            ],
        ), contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as exit_info:
                main()

        self.assertEqual(exit_info.exception.code, 0)
        report = output.getvalue()
        self.assertIn("Raw OCR text:", report)
        self.assertIn(raw_text, report)
        self.assertIn("Candidate 1: Paracetamol", report)
        self.assertIn("Extracted strength: 500 mg", report)
        self.assertIn("Extracted dosage form: tablet", report)
        self.assertIn("Database match status: matched", report)
        self.assertIn("Storage location: DEMO-A1", report)
        self.assertIn("Inventory quantity: 12", report)
        self.assertIn("no robot request was created or dispatched", report)

    def test_missing_strength_is_never_resolved_to_a_catalog_location(self) -> None:
        parsed = parse_prescription_text("Tab Paracetamol")
        self.assertEqual(parsed.candidates[0].name, "Paracetamol")
        self.assertEqual(parsed.candidates[0].strength, None)
        self.assertEqual(parsed.candidates[0].dosage_form, "tablet")
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            return_value="Tab Paracetamol",
        ):
            result = scan_image(self.image_path, self.matcher, self.repository)
        self.assertEqual(
            result.items[0].database_status,
            "incomplete_requires_review",
        )
        self.assertIsNone(result.items[0].location)
        self.assertEqual(result.items[0].review_fields, ("strength",))

    def test_missing_tesseract_is_reported_as_an_ocr_error(self) -> None:
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            side_effect=pytesseract.TesseractNotFoundError(),
        ):
            with self.assertRaises(TesseractUnavailableError):
                scan_image(self.image_path, self.matcher, self.repository)

    def test_ocr_failure_is_reported_and_never_dispatches_a_robot_command(self) -> None:
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            side_effect=pytesseract.TesseractError(1, "OCR failed"),
        ):
            with self.assertRaises(PrescriptionOCRError):
                scan_image(self.image_path, self.matcher, self.repository)

        output = io.StringIO()
        with patch(
            "pharmacy_assistant.prescription.pytesseract.image_to_string",
            side_effect=pytesseract.TesseractError(1, "OCR failed"),
        ), patch(
            "sys.argv",
            [
                "pharmacy-assistant",
                "scan",
                "--image",
                str(self.image_path),
                "--db",
                str(self.database_path),
            ],
        ), contextlib.redirect_stderr(output):
            with self.assertRaises(SystemExit) as exit_info:
                main()

        self.assertEqual(exit_info.exception.code, 1)
        self.assertIn("Tesseract could not read the image", output.getvalue())
        self.assertNotIn("dispatch", output.getvalue().lower())
        self.assertNotIn("robot command", output.getvalue().lower())

    def test_webcam_preview_waits_for_space_then_releases_resources(self) -> None:
        frame = np.full((120, 160, 3), 255, dtype=np.uint8)
        camera = unittest.mock.Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        with patch(
            "pharmacy_assistant.prescription.cv2.VideoCapture",
            return_value=camera,
        ), patch(
            "pharmacy_assistant.prescription.cv2.imshow"
        ) as imshow, patch(
            "pharmacy_assistant.prescription.cv2.waitKey",
            side_effect=[-1, 32],
        ), patch(
            "pharmacy_assistant.prescription.cv2.destroyAllWindows"
        ) as destroy, contextlib.redirect_stdout(io.StringIO()):
            captured = capture_webcam_image(2)

        self.assertIs(captured, frame)
        self.assertEqual(camera.read.call_count, 2)
        self.assertEqual(imshow.call_count, 2)
        self.assertEqual(
            imshow.call_args.args[0],
            "Pharmacy Assistant - Prescription Camera",
        )
        camera.release.assert_called_once()
        destroy.assert_called_once()

    def test_webcam_cancel_releases_camera_and_closes_preview(self) -> None:
        frame = np.full((120, 160, 3), 255, dtype=np.uint8)
        camera = unittest.mock.Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (True, frame)
        with patch(
            "pharmacy_assistant.prescription.cv2.VideoCapture",
            return_value=camera,
        ), patch(
            "pharmacy_assistant.prescription.cv2.imshow"
        ), patch(
            "pharmacy_assistant.prescription.cv2.waitKey",
            return_value=ord("q"),
        ), patch(
            "pharmacy_assistant.prescription.cv2.destroyAllWindows"
        ) as destroy, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(CameraCaptureCancelled):
                capture_webcam_image(0)

        camera.release.assert_called_once()
        destroy.assert_called_once()

    def test_unavailable_camera_and_empty_frame_are_reported_and_cleaned_up(self) -> None:
        camera = unittest.mock.Mock()
        camera.isOpened.return_value = False
        with patch(
            "pharmacy_assistant.prescription.cv2.VideoCapture",
            return_value=camera,
        ), patch(
            "pharmacy_assistant.prescription.cv2.destroyAllWindows"
        ) as destroy:
            with self.assertRaises(CameraCaptureError):
                capture_webcam_image(4)
        camera.release.assert_called_once()
        destroy.assert_called_once()

        camera = unittest.mock.Mock()
        camera.isOpened.return_value = True
        camera.read.return_value = (False, None)
        with patch(
            "pharmacy_assistant.prescription.cv2.VideoCapture",
            return_value=camera,
        ), patch(
            "pharmacy_assistant.prescription.cv2.destroyAllWindows"
        ) as destroy:
            with self.assertRaises(CameraCaptureError):
                capture_webcam_image(1)
        camera.release.assert_called_once()
        destroy.assert_called_once()


if __name__ == "__main__":
    unittest.main()
