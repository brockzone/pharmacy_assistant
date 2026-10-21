# Printed medicine-name OCR evaluation data

This directory is for evaluating Tesseract's reading of printed text, especially
`PARACETAMOL`, `ASPIRIN`, and `AMOXICILLIN`. It is not an image-based medicine
recognizer, a training pipeline, or a dispensing workflow. OCR stays review-only;
strict database matching and the default scanner behavior are not changed.

## Current status

There are no photographs or verified ground-truth labels in this repository
yet. The `evaluation/` directory is intended for the held-out real-photo test
set. Once you add labeled photos, the evaluation script and baseline plus
opt-in experiment comparisons can be created and run. Do not use held-out
evaluation samples to tune OCR settings.

## Evaluation photo layout

Store each photograph with its label in `evaluation/`, using the same basename:

```text
training/
  evaluation/
    sample001.jpg
    sample001.gt.txt
    sample002.png
    sample002.gt.txt
  manifest.csv
```

Use `.png`, `.jpg`, or `.jpeg` image files. The `.gt.txt` file must be UTF-8
plain text containing the exact, human-verified printed text visible in the
photo (including case, punctuation, spacing, strengths, and dosage forms).
Use a final newline. If the photo contains multiple relevant lines, transcribe
all visible text in reading order, one printed line per text line. Do not infer
or correct text that is not legible; exclude an unusable photo rather than
guessing its label.

Evaluation photographs must be real photographs, not generated images.
Photograph printed labels or packaging, not medicine appearance for
classification. Do not include patient names, prescription identifiers, or
other personal or health information.

## Samples to collect

Include diverse print conditions for each target name: different fonts and
sizes, packaging and backgrounds, lighting, glare/shadows, orientation, focus,
and image quality. Include the target names:

- `PARACETAMOL`
- `ASPIRIN`
- `AMOXICILLIN`

Also include negative/look-alike text that must not be read as a target:

- `AMOXAPINE`
- `ASPARTAME`
- `PARACETOL`

Include non-target catalog medicine names, strengths such as `250 mg`,
`500 mg`, and `5 mg/ml`, and forms such as `tablet`, `capsule`, and `syrup`.
Record the actual visible transcription in the matching `.gt.txt` file.

## Manifest and split hygiene

Add one row to `manifest.csv` per physical source photo. Required columns are:

| Column | Meaning |
| --- | --- |
| `source_photo_id` | Stable unique ID for the original physical photo/sample |
| `image_file` | Path relative to `training/`, for example `evaluation/sample001.jpg` |
| `ground_truth_file` | Matching relative `.gt.txt` path |
| `split` | `test` for this held-out evaluation directory |
| `class` | Main printed target, non-target, or look-alike text |
| `conditions` | Relevant font, size, lighting, orientation, packaging, and quality details |
| `license_origin` | Who created/provided the photo, permission or license, and source |

The evaluation directory is reserved for the held-out `test` split. Do not
place a photo there if it or another crop, rotation, brightness variant, or
augmentation of the same physical source has been used for training or tuning.
When training/validation data are later collected, assign splits by
`source_photo_id` first and keep every derivative of that physical photo in the
same split. Never split by filename alone.

For example, a photo of `AMOXAPINE` remains a look-alike negative even if it is
cropped or rotated. Do not relabel negatives as target medicines. Record
permission and origin accurately; do not add samples whose usage rights are
unclear.

## Before evaluation

Verify each photo/label pair manually, ensure the `.gt.txt` transcription is
exact, and complete the manifest. The evaluation should report the exact OCR
output per image, target-name word accuracy and character error rate, and
look-alike false reads. Optional user-word, preprocessing, and additional PSM
experiments must be measured separately from the existing pipeline and remain
opt-in. The existing default OCR and strict matching behavior must not change
unless a separately reviewed change is later approved.
