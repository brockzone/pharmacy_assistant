"""Command-line entry point for Phase 1 database setup and inspection."""

import argparse
import json
import sqlite3
import sys
from typing import TYPE_CHECKING

from pharmacy_assistant.database import connect, initialize_database, seed_demo_data
from pharmacy_assistant.matching import MedicineMatcher
from pharmacy_assistant.repository import MedicineRepository
from pharmacy_assistant.resolver import RetrievalRequestBuilder

if TYPE_CHECKING:
    from pharmacy_assistant.prescription import PrescriptionScanResult


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="pharmacy-assistant")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="initialize the local database")
    init_parser.add_argument("--db", required=True, help="SQLite database path")
    init_parser.add_argument(
        "--demo", action="store_true", help="add fictional test records"
    )

    match_parser = subparsers.add_parser("match", help="match one medicine")
    match_parser.add_argument("--db", required=True, help="SQLite database path")
    match_parser.add_argument("--name", required=True)
    match_parser.add_argument("--strength")
    match_parser.add_argument("--form", dest="dosage_form")

    request_parser = subparsers.add_parser(
        "request", help="build a retrieval request for one medicine"
    )
    request_parser.add_argument("--db", required=True, help="SQLite database path")
    request_parser.add_argument("--name", required=True)
    request_parser.add_argument("--strength")
    request_parser.add_argument("--form", dest="dosage_form")

    scan_parser = subparsers.add_parser(
        "scan", help="scan a prescription image for human review"
    )
    scan_parser.add_argument("--db", required=True, help="SQLite database path")
    scan_source = scan_parser.add_mutually_exclusive_group(required=True)
    scan_source.add_argument("--image", help="JPG, JPEG, or PNG prescription image")
    scan_source.add_argument(
        "--camera", type=int, metavar="INDEX", help="capture one frame from a webcam"
    )
    scan_parser.add_argument(
        "--debug-dir",
        help="optionally save captured/preprocessed images and OCR attempts",
    )
    scan_parser.add_argument(
        "--ocr-model",
        help="optional custom traineddata model name or path (e.g. pharmacy_meds)",
    )

    gui_parser = subparsers.add_parser(
        "gui", help="open the desktop prescription scanner"
    )
    gui_parser.add_argument("--db", required=True, help="SQLite database path")
    gui_parser.add_argument(
        "--camera", type=int, default=0, metavar="INDEX", help="webcam index (default: 0)"
    )
    gui_parser.add_argument(
        "--ocr-model",
        help="optional custom traineddata model name or path (e.g. pharmacy_meds)",
    )

    capture_parser = subparsers.add_parser(
        "capture", help="capture a photograph and enter exact ground-truth label"
    )
    capture_source = capture_parser.add_mutually_exclusive_group()
    capture_source.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="INDEX",
        help="webcam index (default: 0)",
    )
    capture_source.add_argument(
        "--image", help="existing image file to label instead of camera capture"
    )
    capture_parser.add_argument(
        "--split",
        choices=["test", "train", "val"],
        default="test",
        help="dataset split (default: test)",
    )
    capture_parser.add_argument(
        "--source-id",
        help="physical source photo ID (for grouping crops/variants)",
    )
    capture_parser.add_argument(
        "--training-dir",
        default="training",
        help="path to training directory (default: training)",
    )
    return parser


def _print_scan_report(result: "PrescriptionScanResult") -> None:
    print("=== Prescription OCR Scan ===")
    print()
    print("Raw OCR text:")
    print("---")
    print(result.raw_text or "(No text was recognized.)")
    print("---")
    print()

    if result.ocr_configuration is not None:
        print(f"Selected OCR configuration: {result.ocr_configuration}")
        if result.ocr_score is not None:
            print(
                f"OCR selection score (not a confidence probability): "
                f"{result.ocr_score:.2f}"
            )
        print()

    if not result.items:
        print("Medicine candidates: none")
    for index, item in enumerate(result.items, start=1):
        candidate = item.candidate
        print(f"Candidate {index}: {candidate.name or '(medicine name unclear)'}")
        print(f"  Extracted strength: {candidate.strength or '(not identified)'}")
        print(f"  Extracted dosage form: {candidate.dosage_form or '(not identified)'}")
        print(f"  Database match status: {item.database_status}")
        if item.medicine is not None:
            print(
                f"  Matched medicine: "
                f"{item.medicine.generic_name} {item.medicine.strength} "
                f"{item.medicine.dosage_form}"
            )
            if item.medicine.common_uses:
                print(f"  Common uses: {item.medicine.common_uses}")
            if item.medicine.important_notes:
                print(f"  Important note: {item.medicine.important_notes}")
            if item.medicine.information_source:
                print(f"  Information source: {item.medicine.information_source}")
        elif item.reference is not None:
            print(
                f"  Generic ingredient (product details unconfirmed): "
                f"{item.reference.generic_name}"
            )
            print(f"  Common uses: {item.reference.common_uses}")
            print(f"  Important note: {item.reference.important_notes}")
            print(f"  Information source: {item.reference.information_source}")
        if item.location is not None:
            print(f"  Storage location: {item.location.code}")
        if item.quantity_available is not None:
            print(f"  Inventory quantity: {item.quantity_available}")
        if item.review_fields:
            print(f"  Fields requiring human review: {', '.join(item.review_fields)}")
        else:
            print("  Fields requiring human review: none")
        if item.database_status == "not_found":
            print("  No confident database match was found.")
        print()

    if result.unparsed_lines:
        print("Unparsed lines requiring human review:")
        for line in result.unparsed_lines:
            print(f"  - {line}")
        print()

    if not result.items:
        print("No confident medicine match was found; human review is required.")
    if not result.raw_text.strip():
        print("Human review required: OCR returned no readable text.")
    print("Review only: no robot request was created or dispatched.")


def _scan(args: argparse.Namespace) -> int:
    try:
        from pharmacy_assistant.prescription import (
            CameraCaptureCancelled,
            PrescriptionError,
            capture_webcam_image,
            scan_image,
            scan_image_array,
        )
    except ImportError as error:
        print(
            "Prescription scanning dependencies are missing. Install project "
            "dependencies with: python3 -m pip install -e .",
            file=sys.stderr,
        )
        return 1

    try:
        with connect(args.db) as connection:
            matcher = MedicineMatcher(connection)
            repository = MedicineRepository(connection)
            if args.image is not None:
                result = scan_image(
                    args.image,
                    matcher,
                    repository,
                    debug_dir=args.debug_dir,
                    ocr_model=args.ocr_model,
                )
            else:
                image = capture_webcam_image(args.camera)
                result = scan_image_array(
                    image,
                    matcher,
                    repository,
                    debug_dir=args.debug_dir,
                    ocr_model=args.ocr_model,
                )
    except CameraCaptureCancelled as error:
        print(error)
        return 0
    except (PrescriptionError, sqlite3.Error) as error:
        print(f"Prescription scan failed: {error}", file=sys.stderr)
        return 1

    _print_scan_report(result)
    return 0


def main() -> None:
    args = _parser().parse_args()
    if args.command == "init":
        if args.demo:
            seed_demo_data(args.db)
        else:
            initialize_database(args.db)
        print(f"Initialized database: {args.db}")
        return

    if args.command == "scan":
        raise SystemExit(_scan(args))

    if args.command == "capture":
        from pharmacy_assistant.capture import capture_and_label_interactive

        capture_and_label_interactive(
            camera_index=args.camera,
            image_path=args.image,
            split=args.split,
            source_photo_id=args.source_id,
            training_root=args.training_dir,
        )
        return

    if args.command == "gui":
        from pharmacy_assistant.gui import launch_dashboard

        launch_dashboard(args.db, args.camera, args.ocr_model)
        return

    with sqlite3.connect(args.db) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if args.command == "match":
            result = MedicineMatcher(connection).match(
                args.name, args.strength, args.dosage_form
            )
            print(
                json.dumps(
                    {
                        "status": result.status.value,
                        "candidates": [
                            {
                                "id": medicine.id,
                                "generic_name": medicine.generic_name,
                                "brand_name": medicine.brand_name,
                                "strength": medicine.strength,
                                "dosage_form": medicine.dosage_form,
                            }
                            for medicine in result.candidates
                        ],
                    },
                    indent=2,
                )
            )
            return

        request = RetrievalRequestBuilder(connection).build(
            [
                {
                    "name": args.name,
                    "strength": args.strength,
                    "dosage_form": args.dosage_form,
                }
            ]
        )
        print(
            json.dumps(
                {
                    "request_id": request.request_id,
                    "items": [
                        {
                            "requested_name": item.requested_name,
                            "requested_strength": item.requested_strength,
                            "requested_dosage_form": item.requested_dosage_form,
                            "status": item.status.value,
                            "medicine_id": item.medicine.id if item.medicine else None,
                            "location": (
                                {
                                    "code": item.location.code,
                                    "x": item.location.x,
                                    "y": item.location.y,
                                    "yaw": item.location.yaw,
                                }
                                if item.location
                                else None
                            ),
                            "quantity_available": item.quantity_available,
                            "candidate_medicine_ids": item.candidate_medicine_ids,
                            "message": item.message,
                        }
                        for item in request.items
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
