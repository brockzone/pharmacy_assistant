"""Evaluation script for baseline OCR and cheap fixes on held-out photos."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import pytesseract

# Ensure project src is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pharmacy_assistant.database import connect
from pharmacy_assistant.matching import MedicineMatcher
from pharmacy_assistant.prescription import (
    PrescriptionScanResult,
    _ocr_text_score,
    parse_prescription_text,
    preprocess_image,
    scan_image,
)
from pharmacy_assistant.repository import MedicineRepository

LOOKALIKE_CLASSES = {"AMOXAPINE", "ASPARTAME", "PARACETOL"}
TARGET_CLASSES = {"PARACETAMOL", "ASPIRIN", "AMOXICILLIN"}


def edit_distance(seq1: list | str, seq2: list | str) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq1[i - 1] == seq2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[m][n]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate CER = edit_distance(ref, hyp) / len(ref)."""
    ref_norm = " ".join(reference.strip().split())
    hyp_norm = " ".join(hypothesis.strip().split())
    if not ref_norm:
        return 0.0 if not hyp_norm else 1.0
    return edit_distance(ref_norm, hyp_norm) / len(ref_norm)


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Calculate WER = word_edit_distance(ref, hyp) / len(ref_words)."""
    ref_words = reference.strip().split()
    hyp_words = hypothesis.strip().split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return edit_distance(ref_words, hyp_words) / len(ref_words)


@dataclass
class EvalResult:
    sample_id: str
    target_class: str
    ground_truth: str
    raw_ocr_text: str
    extracted_candidate: str | None
    cer: float
    wer: float
    is_target_hit: bool
    is_lookalike_false_match: bool


def _run_baseline_ocr(
    image_path: Path,
    matcher: MedicineMatcher,
    repository: MedicineRepository,
) -> tuple[str, str | None, bool, bool]:
    """Run existing baseline prescription OCR."""
    result: PrescriptionScanResult = scan_image(image_path, matcher, repository)
    raw = result.raw_text.strip()
    candidate = result.items[0].candidate.name if result.items else None
    return raw, candidate, bool(result.items), False


def _run_extra_psm_ocr(
    image_path: Path,
    matcher: MedicineMatcher,
    repository: MedicineRepository,
) -> tuple[str, str | None, bool, bool]:
    """Run OCR adding PSM 3 (fully automatic) and PSM 1 (with OSD) to recipes."""
    image = cv2.imread(str(image_path))
    variants = preprocess_image(image)
    recipes = (
        ("original", 7),
        ("grayscale", 7),
        ("otsu", 7),
        ("adaptive", 7),
        ("grayscale", 6),
        ("grayscale", 11),
        ("original", 3),
        ("grayscale", 3),
        ("otsu", 3),
        ("adaptive", 3),
        ("original", 1),
    )
    best_text = ""
    best_score = -1.0
    for vname, psm in recipes:
        config = f"--oem 3 --psm {psm}"
        try:
            txt = pytesseract.image_to_string(variants[vname], config=config)
        except Exception:
            continue
        score = _ocr_text_score(txt, matcher)
        if score > best_score:
            best_score = score
            best_text = txt

    parsed = parse_prescription_text(best_text, matcher)
    candidate = parsed.candidates[0].name if parsed.candidates else None
    return best_text.strip(), candidate, bool(parsed.candidates), False


def _run_preprocessing_tweak_ocr(
    image_path: Path,
    matcher: MedicineMatcher,
    repository: MedicineRepository,
) -> tuple[str, str | None, bool, bool]:
    """Run OCR with tweaked preprocessing: raw color, color channels, and selective denoising."""
    image = cv2.imread(str(image_path))
    variants = preprocess_image(image)
    # Add raw color image and high-contrast green channel
    variants["raw_bgr"] = image
    if image.ndim == 3 and image.shape[2] == 3:
        variants["channel_g"] = image[:, :, 1]
        variants["channel_b"] = image[:, :, 0]

    recipes = (
        ("raw_bgr", 3),
        ("channel_g", 3),
        ("channel_b", 3),
        ("original", 7),
        ("grayscale", 7),
        ("otsu", 7),
        ("adaptive", 7),
        ("grayscale", 6),
        ("grayscale", 11),
        ("original", 3),
        ("grayscale", 3),
        ("otsu", 3),
    )
    best_text = ""
    best_score = -1.0
    for vname, psm in recipes:
        if vname not in variants:
            continue
        config = f"--oem 3 --psm {psm}"
        try:
            txt = pytesseract.image_to_string(variants[vname], config=config)
        except Exception:
            continue
        score = _ocr_text_score(txt, matcher)
        if score > best_score:
            best_score = score
            best_text = txt

    parsed = parse_prescription_text(best_text, matcher)
    candidate = parsed.candidates[0].name if parsed.candidates else None
    return best_text.strip(), candidate, bool(parsed.candidates), False


def _run_user_words_ocr(
    image_path: Path,
    matcher: MedicineMatcher,
    repository: MedicineRepository,
) -> tuple[str, str | None, bool, bool]:
    """Run baseline OCR but with user-words dictionary flag."""
    user_words_path = PROJECT_ROOT / "training" / "user_words.txt"
    image = cv2.imread(str(image_path))
    variants = preprocess_image(image)
    recipes = (
        ("original", 7),
        ("grayscale", 7),
        ("otsu", 7),
        ("adaptive", 7),
        ("grayscale", 6),
        ("grayscale", 11),
    )
    best_text = ""
    best_score = -1.0
    for vname, psm in recipes:
        config = f"--oem 3 --psm {psm} --user-words {user_words_path}"
        try:
            txt = pytesseract.image_to_string(variants[vname], config=config)
        except Exception:
            continue
        score = _ocr_text_score(txt, matcher)
        if score > best_score:
            best_score = score
            best_text = txt

    parsed = parse_prescription_text(best_text, matcher)
    candidate = parsed.candidates[0].name if parsed.candidates else None
    return best_text.strip(), candidate, bool(parsed.candidates), False


def _run_combined_cheap_fixes(
    image_path: Path,
    matcher: MedicineMatcher,
    repository: MedicineRepository,
) -> tuple[str, str | None, bool, bool]:
    """Combine: Extra PSM modes (3, 1) + Preprocessing tweaks (color channels) + User-words."""
    user_words_path = PROJECT_ROOT / "training" / "user_words.txt"
    image = cv2.imread(str(image_path))
    variants = preprocess_image(image)
    variants["raw_bgr"] = image
    if image.ndim == 3 and image.shape[2] == 3:
        variants["channel_g"] = image[:, :, 1]
        variants["channel_b"] = image[:, :, 0]

    recipes = (
        ("raw_bgr", 3),
        ("channel_g", 3),
        ("channel_b", 3),
        ("original", 3),
        ("grayscale", 3),
        ("otsu", 3),
        ("adaptive", 3),
        ("original", 7),
        ("grayscale", 7),
        ("grayscale", 6),
        ("grayscale", 11),
    )
    best_text = ""
    best_score = -1.0
    for vname, psm in recipes:
        if vname not in variants:
            continue
        config = f"--oem 3 --psm {psm} --user-words {user_words_path}"
        try:
            txt = pytesseract.image_to_string(variants[vname], config=config)
        except Exception:
            continue
        score = _ocr_text_score(txt, matcher)
        if score > best_score:
            best_score = score
            best_text = txt

    parsed = parse_prescription_text(best_text, matcher)
    candidate = parsed.candidates[0].name if parsed.candidates else None
    return best_text.strip(), candidate, bool(parsed.candidates), False


def _run_custom_model_ocr(
    image_path: Path,
    matcher: MedicineMatcher,
    repository: MedicineRepository,
) -> tuple[str, str | None, bool, bool]:
    """Run OCR using the fine-tuned pharmacy_meds.traineddata model."""
    custom_tessdata = PROJECT_ROOT / "training"
    image = cv2.imread(str(image_path))
    variants = preprocess_image(image)
    variants["raw_bgr"] = image
    if image.ndim == 3 and image.shape[2] == 3:
        variants["channel_g"] = image[:, :, 1]
        variants["channel_b"] = image[:, :, 0]

    recipes = (
        ("raw_bgr", 3),
        ("channel_g", 3),
        ("channel_b", 3),
        ("original", 3),
        ("grayscale", 3),
        ("otsu", 3),
        ("adaptive", 3),
        ("original", 7),
        ("grayscale", 7),
    )
    best_text = ""
    best_score = -1.0
    for vname, psm in recipes:
        if vname not in variants:
            continue
        config = f"--tessdata-dir {custom_tessdata} -l pharmacy_meds --oem 1 --psm {psm}"
        try:
            txt = pytesseract.image_to_string(variants[vname], config=config)
        except Exception:
            continue
        score = _ocr_text_score(txt, matcher)
        if score > best_score:
            best_score = score
            best_text = txt

    parsed = parse_prescription_text(best_text, matcher)
    candidate = parsed.candidates[0].name if parsed.candidates else None
    return best_text.strip(), candidate, bool(parsed.candidates), False


PIPELINE_RUNNERS = {
    "Baseline": _run_baseline_ocr,
    "Fix 1: Extra PSM": _run_extra_psm_ocr,
    "Fix 2: Preprocessing Tweaks": _run_preprocessing_tweak_ocr,
    "Fix 3: User Words": _run_user_words_ocr,
    "Fix 4: Combined Fixes": _run_combined_cheap_fixes,
    "Custom Model (pharmacy_meds)": _run_custom_model_ocr,
}


def evaluate_dataset(
    manifest_path: Path,
    split: str = "test",
    db_path: Path | str = "pharmacy.db",
) -> dict[str, list[EvalResult]]:
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    training_dir = manifest_path.parent
    records = []
    with manifest_path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["split"] == split:
                records.append(row)

    if not records:
        print(f"Warning: No samples found in manifest for split={split}.")
        return {}

    with connect(str(db_path)) as connection:
        matcher = MedicineMatcher(connection)
        repository = MedicineRepository(connection)

        all_results: dict[str, list[EvalResult]] = {}

        for mode_name, runner_fn in PIPELINE_RUNNERS.items():
            results = []
            for row in records:
                sample_id = row["source_photo_id"]
                target_cls = row["class"].strip()
                img_path = training_dir / row["image_file"]
                gt_path = training_dir / row["ground_truth_file"]

                gt_text = gt_path.read_text(encoding="utf-8").strip()

                raw_ocr, candidate, has_candidate, _ = runner_fn(
                    img_path, matcher, repository
                )

                cer = character_error_rate(gt_text, raw_ocr)
                wer = word_error_rate(gt_text, raw_ocr)

                # Target hit: if target class is recognized
                is_hit = False
                if target_cls in TARGET_CLASSES:
                    if candidate and candidate.upper() == target_cls.upper():
                        is_hit = True
                    elif target_cls.upper() in raw_ocr.upper():
                        is_hit = True

                # False match check on look-alikes
                is_false_match = False
                if target_cls in LOOKALIKE_CLASSES:
                    if candidate and candidate.upper() in TARGET_CLASSES:
                        is_false_match = True

                results.append(
                    EvalResult(
                        sample_id=sample_id,
                        target_class=target_cls,
                        ground_truth=gt_text,
                        raw_ocr_text=raw_ocr,
                        extracted_candidate=candidate,
                        cer=cer,
                        wer=wer,
                        is_target_hit=is_hit,
                        is_lookalike_false_match=is_false_match,
                    )
                )
            all_results[mode_name] = results

    return all_results


def print_evaluation_report(all_results: dict[str, list[EvalResult]]) -> None:
    print("\n" + "=" * 80)
    print("STEP 3 - OCR EVALUATION REPORT (HELD-OUT TEST SAMPLES)")
    print("=" * 80)

    for mode_name, results in all_results.items():
        print(f"\n### Pipeline Configuration: {mode_name}")
        print("-" * 80)
        print(
            f"{'Sample ID':<22} | {'Class':<12} | {'CER':<6} | {'WER':<6} | {'Candidate':<16} | {'Raw OCR (first 30 chars)'}"
        )
        print("-" * 80)
        for r in results:
            preview = repr(r.raw_ocr_text[:30].replace("\n", " "))
            cand_str = r.extracted_candidate or "(None)"
            print(
                f"{r.sample_id:<22} | {r.target_class:<12} | {r.cer:<6.2f} | {r.wer:<6.2f} | {cand_str:<16} | {preview}"
            )

    print("\n" + "=" * 80)
    print("SUMMARY COMPARISON ACROSS CHEAP FIXES")
    print("=" * 80)
    print(
        f"{'Configuration':<28} | {'Avg CER':<8} | {'Avg WER':<8} | {'Target Recall':<14} | {'Lookalike False Matches'}"
    )
    print("-" * 80)

    for mode_name, results in all_results.items():
        avg_cer = sum(r.cer for r in results) / len(results)
        avg_wer = sum(r.wer for r in results) / len(results)
        targets = [r for r in results if r.target_class in TARGET_CLASSES]
        target_hits = sum(1 for r in targets if r.is_target_hit)
        recall_str = f"{target_hits}/{len(targets)}" if targets else "N/A"
        false_matches = sum(1 for r in results if r.is_lookalike_false_match)
        print(
            f"{mode_name:<28} | {avg_cer:<8.2%} | {avg_wer:<8.2%} | {recall_str:<14} | {false_matches}"
        )
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OCR on held-out photos.")
    parser.add_argument(
        "--manifest",
        default="training/manifest.csv",
        help="Path to manifest.csv (default: training/manifest.csv)",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--db",
        default="pharmacy.db",
        help="Database path (default: pharmacy.db)",
    )
    args = parser.parse_args()

    results = evaluate_dataset(args.manifest, split=args.split, db_path=args.db)
    if results:
        print_evaluation_report(results)


if __name__ == "__main__":
    main()
