"""Webcam capture and interactive ground-truth labeling for OCR evaluation."""

import csv
from datetime import datetime
from pathlib import Path
import re
import sys
from typing import TextIO

import cv2

from pharmacy_assistant.prescription import (
    CameraCaptureCancelled,
    CameraCaptureError,
    PrescriptionImageError,
    capture_webcam_image,
    load_valid_image,
)

MANIFEST_HEADER = (
    "source_photo_id",
    "image_file",
    "ground_truth_file",
    "split",
    "class",
    "conditions",
    "license_origin",
)


def next_sample_basename(target_dir: Path, prefix: str = "sample") -> str:
    """Generate a clean sequential sample basename like sample001, sample002."""
    target_dir.mkdir(parents=True, exist_ok=True)
    existing = list(target_dir.glob(f"{prefix}[0-9][0-9][0-9].*"))
    indices = []
    for path in existing:
        match = re.search(rf"{prefix}(\d{{3}})", path.stem)
        if match:
            indices.append(int(match.group(1)))
    next_index = max(indices, default=0) + 1
    return f"{prefix}{next_index:03d}"


def save_labeled_sample(
    image,
    ground_truth_text: str,
    target_class: str,
    conditions: str,
    split: str,
    source_photo_id: str | None,
    license_origin: str,
    training_root: Path,
    sample_name: str | None = None,
) -> tuple[Path, Path, str]:
    """Save the image, its exact .gt.txt label, and append a record to manifest.csv."""
    clean_text = ground_truth_text.strip()
    if not clean_text:
        raise ValueError("Ground-truth text cannot be empty. Never guess a label.")

    training_root = Path(training_root).resolve()
    manifest_path = training_root / "manifest.csv"

    # Evaluation split goes into training/evaluation/; other splits go into training/data/
    if split == "test":
        dest_dir = training_root / "evaluation"
    else:
        dest_dir = training_root / "data"
    dest_dir.mkdir(parents=True, exist_ok=True)

    if not sample_name:
        sample_name = next_sample_basename(dest_dir)

    image_path = dest_dir / f"{sample_name}.png"
    gt_path = dest_dir / f"{sample_name}.gt.txt"

    if not cv2.imwrite(str(image_path), image):
        raise PrescriptionImageError(f"Could not save captured image to {image_path}")

    # Write exact ground-truth text with a trailing newline
    gt_path.write_text(clean_text + "\n", encoding="utf-8")

    # If no source_photo_id is provided, treat this physical photo as its own unique source
    actual_source_id = source_photo_id.strip() if source_photo_id else sample_name

    # Paths in manifest.csv are relative to training/
    rel_image = str(image_path.relative_to(training_root))
    rel_gt = str(gt_path.relative_to(training_root))

    # Ensure manifest header exists if file is empty
    file_exists = manifest_path.is_file()
    write_header = not file_exists or manifest_path.stat().st_size == 0

    with manifest_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(MANIFEST_HEADER)
        writer.writerow(
            [
                actual_source_id,
                rel_image,
                rel_gt,
                split,
                target_class.strip(),
                conditions.strip(),
                license_origin.strip(),
            ]
        )

    return image_path, gt_path, actual_source_id


def capture_and_label_interactive(
    camera_index: int = 0,
    image_path: Path | str | None = None,
    split: str = "test",
    source_photo_id: str | None = None,
    training_root: Path | str = "training",
    input_stream: TextIO | None = None,
) -> tuple[Path, Path, str] | None:
    """Interactively capture a photo or load an existing one, then prompt for ground-truth."""
    training_path = Path(training_root).resolve()

    if image_path is not None:
        image = load_valid_image(image_path)
        print(f"Loaded existing image: {image_path}")
    else:
        try:
            image = capture_webcam_image(camera_index)
        except CameraCaptureCancelled:
            print("Capture cancelled by operator.")
            return None

    prompt_stream = input_stream or sys.stdin

    def _prompt(question: str, default: str = "") -> str:
        prompt_text = f"{question} [{default}]: " if default else f"{question}: "
        print(prompt_text, end="", flush=True)
        line = prompt_stream.readline()
        if not line:
            return default
        answer = line.strip()
        return answer if answer else default

    print("\n--- Ground-Truth Labeling ---")
    print("Transcribe ONLY the exact visible printed text on this sample.")
    print("Include strengths (e.g. 500 mg) and dosage forms (e.g. tablet) if visible.")
    print("If multiple lines are visible, you can separate them with a slash or newline.")

    gt_text = ""
    while not gt_text:
        gt_text = _prompt("Exact visible text (required, never guess)")
        if not gt_text:
            print("Error: Ground truth cannot be empty. Re-enter or Ctrl+C to cancel.")

    target_class = _prompt(
        "Target medicine or class (e.g. PARACETAMOL, ASPIRIN, AMOXICILLIN, AMOXAPINE)",
        default=gt_text.split()[0].upper() if gt_text else "UNKNOWN",
    )
    conditions = _prompt(
        "Lighting & packaging conditions (e.g. blister pack, white paper, overhead LED, glare)",
        default="webcam_capture, indoor_light",
    )
    prompted_source_id = _prompt(
        "Physical source photo ID (press Enter for auto-id, or enter ID if this is a crop/variant of an existing photo)",
        default=source_photo_id or "",
    )
    origin = _prompt("License / Origin", default="operator_webcam")

    img_saved, gt_saved, final_source_id = save_labeled_sample(
        image=image,
        ground_truth_text=gt_text,
        target_class=target_class,
        conditions=conditions,
        split=split,
        source_photo_id=prompted_source_id or None,
        license_origin=origin,
        training_root=training_path,
    )

    print("\n[SUCCESS] Sample captured and registered:")
    print(f"  - Image: {img_saved}")
    print(f"  - Ground Truth: {gt_saved}")
    print(f"  - Source Photo ID: {final_source_id}")
    print(f"  - Split: {split}")
    print(f"  - Class: {target_class}")
    return img_saved, gt_saved, final_source_id
