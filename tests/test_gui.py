"""Tests for scanner UI state and aspect-preserving image presentation."""

import queue
import unittest
from datetime import datetime
from unittest.mock import Mock, patch

import numpy as np

from pharmacy_assistant.gui import CameraWorker, PharmacyDashboard, ScannerState
from pharmacy_assistant.prescription import (
    PrescriptionScanResult,
    preprocess_image,
)


class ScannerStateTests(unittest.TestCase):
    def test_live_frames_do_not_replace_persistent_capture(self) -> None:
        state = ScannerState()
        first_live_frame = np.full((120, 160, 3), 30, dtype=np.uint8)
        captured_gray = np.full((120, 160), 180, dtype=np.uint8)
        second_live_frame = np.full((120, 160, 3), 90, dtype=np.uint8)

        state.update_live(first_live_frame)
        generation = state.begin_capture(datetime.now())
        self.assertTrue(state.set_captured_image(generation, captured_gray))
        state.update_live(second_live_frame)

        self.assertIs(state.live_frame, second_live_frame)
        self.assertIs(state.captured_image, captured_gray)
        self.assertEqual(int(state.captured_image[0, 0]), 180)

    def test_clear_scan_removes_capture_and_invalidates_old_results(self) -> None:
        state = ScannerState()
        generation = state.begin_capture(datetime.now())
        image = np.full((100, 100), 200, dtype=np.uint8)
        self.assertTrue(state.set_captured_image(generation, image))

        state.clear()

        self.assertIsNone(state.captured_image)
        self.assertIsNone(state.captured_at)
        self.assertIsNone(state.result)
        self.assertFalse(state.set_captured_image(generation, image))
        self.assertFalse(
            state.set_result(generation, PrescriptionScanResult("", (), ()))
        )

    def test_scan_error_state_rejects_stale_error(self) -> None:
        state = ScannerState()
        generation = state.begin_capture(datetime.now())

        self.assertTrue(state.set_error(generation))
        self.assertEqual(state.status, "error")
        state.clear()
        self.assertFalse(state.set_error(generation))

    def test_camera_open_failure_is_reported_and_releases_resource(self) -> None:
        camera = Mock()
        camera.isOpened.return_value = False
        frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        events: queue.Queue[tuple[str, object]] = queue.Queue()
        worker = CameraWorker(3, frames, events)

        with patch("pharmacy_assistant.gui.cv2.VideoCapture", return_value=camera):
            worker.run()

        self.assertEqual(
            events.get_nowait(),
            ("camera_error", "Could not open camera 3."),
        )
        camera.release.assert_called_once()

    def test_match_status_colors_never_treat_review_as_confirmed(self) -> None:
        self.assertEqual(
            PharmacyDashboard._overall_status(["matched"]),
            ("Database match", "#EAF7EE", "#16803C"),
        )
        self.assertEqual(
            PharmacyDashboard._overall_status(["ambiguous"]),
            ("Review required", "#FFF7E6", "#A05A00"),
        )
        self.assertEqual(
            PharmacyDashboard._overall_status(["not_found"]),
            ("No reliable match", "#FEF0EF", "#B42318"),
        )

    def test_processed_grayscale_is_an_existing_ocr_variant(self) -> None:
        image = np.zeros((120, 240, 3), dtype=np.uint8)
        image[:, :120] = (255, 0, 0)
        grayscale = preprocess_image(image)["grayscale"]

        self.assertEqual(grayscale.ndim, 2)
        self.assertEqual(grayscale.dtype, np.uint8)
        self.assertGreater(grayscale.shape[0], image.shape[0])


if __name__ == "__main__":
    unittest.main()
