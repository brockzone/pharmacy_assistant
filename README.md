# Pharmacy Assistant — Phase 2

Phase 1 provides a local SQLite medicine catalog, inventory and storage
locations, strict medicine matching, location resolution, and structured
retrieval requests. Phase 2 adds prescription image OCR and conservative
medicine-candidate matching for human review. It does not communicate with
ROS 2 or control robot hardware.

All records and location coordinates created by `demo` are fictional test data.
No medicine is substituted when identification is missing or ambiguous.
Prescription OCR results are review-only and never create or dispatch robot
requests.

## Project layout

```text
src/pharmacy_assistant/
  cli.py          Initialize, inspect, and scan prescriptions
  gui.py          Responsive desktop prescription scanner
  database.py     SQLite schema, connection setup, and demo records
  matching.py     Exact normalized-name, strength, and form matching
  models.py       Typed records and retrieval-request result types
  repository.py  Catalog and inventory queries
  resolver.py    Medicine-to-storage-location resolution
  prescription.py Image validation, preprocessing, OCR, and candidate matching
tests/
  test_phase1.py  Automated Phase 1 tests
  test_phase2.py  Prescription scanning and OCR tests
training/
  README.md       OCR evaluation data and labeling instructions
  manifest.csv   Source-photo provenance and split manifest
  evaluation/    Held-out real photographs and matching .gt.txt labels
```

## Install

Python 3.10 or newer is required. Install Python dependencies in a virtual
environment:

```bash
cd ~/pharmacy_assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The Python OCR dependency uses the separate Tesseract executable. On Debian or
Ubuntu, install it with:

```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

## Run

Initialize a database with fictional demo data:

```bash
pharmacy-assistant init --db pharmacy.db --demo
```

Scan an image file (JPG, JPEG, or PNG):

```bash
pharmacy-assistant scan --image /path/to/prescription.png --db pharmacy.db
```

Optionally pass `--ocr-model` to use a fine-tuned model (e.g. `pharmacy_meds`):

```bash
pharmacy-assistant scan --image /path/to/prescription.png --db pharmacy.db --ocr-model pharmacy_meds
```

Open a live webcam preview and scan a medicine name from paper:

```bash
PYTHONPATH=src python3 -m pharmacy_assistant.cli scan --db pharmacy.db --camera 0
```

Open the desktop prescription-scanning dashboard:

```bash
PYTHONPATH=src python3 -m pharmacy_assistant.cli gui --db pharmacy.db --camera 0
```

The dashboard uses Python's built-in Tkinter UI and the existing OpenCV,
Tesseract, and SQLite pipeline. The camera preview remains live in the central
panel, while the captured enhanced-grayscale OCR input and structured scan
results are shown separately in the right panel. Capture, Retake, and Clear
Scan control only the current review session. A camera, OCR, or database error
is shown in the interface; no inventory or dispensing action is performed.

The optional custom model can be selected with `--ocr-model pharmacy_meds`.

For the CLI camera preview, place the paper with the medicine name in view.
Press **SPACE** to capture and scan the displayed frame. Press **ESC** or **Q**
to cancel. Select another camera by changing `--camera 0` to the device index
(for example, `--camera 1`). The webcam is released and preview windows are
closed after capture, cancellation, or camera errors.

The scan displays raw OCR text, medicine candidates, extracted strength and
dosage form when recognized, database match status, verified storage location
and inventory quantity when available, and fields requiring human review.
Ambiguous or incomplete candidates require review; a missing or ambiguous
database match is not substituted with another medicine. Scanning never creates
a retrieval request, dispenses medication, or changes inventory.

OCR tries the original grayscale image plus enhanced grayscale, Otsu, and
adaptive-threshold images with Tesseract page segmentation modes (6, 7, and 11),
then ranks the results using legibility and exact catalog evidence. The
displayed selection score is a ranking score,
not a probability or guarantee of OCR correctness. Exact matching and manual
review requirements are unchanged.

### Printed medicine-name OCR evaluation

The OCR evaluation scaffold and instructions are in
[training/README.md](training/README.md).
The capture and evaluation tools are available for collecting verified labels
and comparing OCR configurations. Keep photographs, crops, and derived
variants from one physical source together in the same split. Do not use
held-out evaluation photos to tune OCR options.

No custom OCR options are enabled by default. Evaluation experiments must remain
opt-in and must not change strict medicine matching or review-only behavior.

The fictional `--demo` catalog includes paracetamol, aspirin, and amoxicillin
reference entries. A confirmed match displays the generic name, common uses,
important notes, and the linked MedlinePlus drug information source. These
summaries are general information only, not a diagnosis or treatment
recommendation. If the generic name is recognized but strength or dosage form
is missing or ambiguous, the dashboard may show ingredient-level information
while clearly marking the product details for verification. Aspirin has
bleeding risks and should not be started for
cardiovascular prevention without healthcare-professional advice; antibiotics
such as amoxicillin are used only for appropriate bacterial infections.
For an existing local demo database, rerun
`PYTHONPATH=src python3 -m pharmacy_assistant.cli init --db pharmacy.db --demo`
to add the new catalog entries and reference table without duplicating seeded
records.

To save the captured image, selected preprocessed image, and text/configuration
from each OCR attempt for troubleshooting, pass `--debug-dir`:

```bash
PYTHONPATH=src python3 -m pharmacy_assistant.cli scan \
  --db pharmacy.db --camera 0 --debug-dir ./ocr-debug
```

The same option works with `--image`. Without `--debug-dir`, no debug images or
attempt logs are written.

### Webcam troubleshooting

- If the camera cannot be opened, check that it is connected, that no other
  application is using it, and try another camera index (`--camera 1`).
- On Linux, check that the user has permission to access the camera device.
- If the preview cannot open in a desktop session, make sure OpenCV's GUI
  support and a graphical display session are available. A headless session
  cannot show an interactive preview; use `--image` instead.
- If OCR returns no readable text, improve lighting, hold the paper steady,
  keep the medicine name in focus, fill more of the frame with the text, and
  retake the image. Avoid shadows and glare. Handwriting and
  low-quality images may require manual review.
- If Tesseract is unavailable, install the system Tesseract executable:

  ```bash
  sudo apt-get update
  sudo apt-get install tesseract-ocr
  ```

## Test

Run Phase 1 and Phase 2 unit tests from the project root:

```bash
python -m unittest discover -s tests -v
```

The schema enables SQLite foreign-key enforcement. Inventory quantities cannot
be negative, and a medicine must be matched by name (generic, brand, or alias);
provided strength and dosage form must also match exactly after normalization.
A match with multiple catalog records is marked ambiguous for operator review.
