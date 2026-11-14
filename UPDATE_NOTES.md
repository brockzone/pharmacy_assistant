# Pharmacy Assistant Update Notes

## Printed medicine-name OCR evaluation, fine-tuning, and integration

- Collected real webcam and held-out photographs for target medicine names (`PARACETAMOL`, `ASPIRIN`, `AMOXICILLIN`) documented with exact `.gt.txt` ground truth in `training/evaluation/` and indexed in `training/manifest.csv`.
- Implemented `pharmacy-assistant capture` CLI subcommand with human-in-the-loop ground-truth validation to acquire labeled test and training samples without guessing.
- Built automated evaluation pipeline `training/evaluate.py` tracking Character Error Rate (CER), Word Error Rate (WER), target recall, and look-alike false-positive rates across baseline, heuristics, and custom models.
- Conducted controlled Tesseract LSTM fine-tuning using float base model `tessdata_best/eng.traineddata` and synthesized multi-font training sets generated via official `text2image`, outputting `training/pharmacy_meds.traineddata` (15 MB, error rate 12.244%).
- Integrated optional `--ocr-model` parameter into `pharmacy-assistant scan` (`--ocr-model pharmacy_meds`), preserving stock Tesseract behavior by default.
- Maintained strict Phase 1 matching requirements: OCR outputs remain review-only and never mutate inventory or issue robot commands. All 28 unit tests pass cleanly.

## Printed medicine-name OCR evaluation scaffold

- Added instructions and an empty manifest for collecting real, provenance-
  documented OCR evaluation photos with verified `.gt.txt` labels.
- No evaluation images are present yet. Baseline measurements and optional
  user-words, preprocessing, and page-segmentation experiments are deferred
  until labeled photos are supplied.
- No OCR behavior or medicine-matching rules were changed; defaults remain
  unchanged and scanning remains review-only.

## Prescription OCR and webcam scanning

- Improved OpenCV preprocessing with grayscale conversion, upscaling for small
  text, light denoising, contrast enhancement, and separate Otsu/adaptive
  threshold variants. The original upscaled grayscale image remains available
  as an OCR alternative.
- Added OCR comparison using Tesseract page segmentation modes 6, 7, and 11.
  Results are ranked by readable text and exact evidence from the medicine
  catalog. The reported score is a ranking score, not a confidence probability.
- Improved cleanup of OCR punctuation and spacing. Split medicine-name tokens
  are only joined when the result exactly matches the existing catalog; strength
  text is kept separate. Existing strict matching and human-review rules remain
  in effect.
- Added camera-preview guidance for framing, lighting, shadows/glare, and
  camera steadiness.
- Added optional `--debug-dir` output containing `captured.png`,
  `preprocessed.png`, and `ocr-attempts.txt`. Nothing is saved unless this
  option is supplied.
- The scanner remains review-only. It does not issue robot commands, dispense
  medication, or change inventory.

## Verification

- The Phase 1 and Phase 2 unit tests passed: 23 tests total.
- A generated printed-text image was successfully recognized as
  `PARACETAMOL 500 mg` with the installed Tesseract executable and matched
  against the demo catalog.
- Webcam behavior is covered with mocked camera tests. No physical webcam or
  real camera photograph was used for verification.

## Running the scanner

```bash
PYTHONPATH=src python3 -m pharmacy_assistant.cli scan --db pharmacy.db --camera 0
```

Save diagnostic artifacts while scanning:

```bash
PYTHONPATH=src python3 -m pharmacy_assistant.cli scan \
  --db pharmacy.db --camera 0 --debug-dir ./ocr-debug
```

Run all tests:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```
