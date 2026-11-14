"""
Prepare .box + .lstmf training files using the official tesstrain approach.

What happens (step by step):
1. For each training/val image, generate_line_box.py creates a .box file —
   a plain-text file listing every character in the ground truth with
   bounding coordinates (we use full-image bounds since text2image isn't used).
2. tesseract is then called with the system lstm.train config which reads
   the .box file and produces a binary .lstmf feature file.
3. train_list.txt and val_list.txt are written so lstmtraining knows what to load.
"""
import csv
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
TESSDATA_BEST = TRAINING_DIR / "tessdata_best"
TESSTRAIN = TRAINING_DIR / "tesstrain"
SYSTEM_TESSDATA = Path("/usr/share/tesseract-ocr/4.00/tessdata")
TRAIN_DIR = TRAINING_DIR / "train"
VAL_DIR = TRAINING_DIR / "val"
TRAIN_DIR.mkdir(exist_ok=True)
VAL_DIR.mkdir(exist_ok=True)

# ── 1. Split manifest by physical source_photo_id ────────────────────────────
manifest = list(csv.DictReader((TRAINING_DIR / "manifest.csv").open(encoding="utf-8")))
HELD_OUT_TEST_IDS = {"whatsapp_photo_amox", "whatsapp_photo_asp", "whatsapp_photo_para"}

source_rows: dict[str, list[dict]] = {}
for row in manifest:
    sid = row["source_photo_id"].strip()
    source_rows.setdefault(sid, []).append(row)

train_rows, val_rows, test_rows = [], [], []
for sid, rows in source_rows.items():
    if sid in HELD_OUT_TEST_IDS:
        test_rows.extend(rows)
    elif sid == "1":
        train_rows.extend(rows[:3])
        val_rows.extend(rows[3:])
    elif sid == "2":
        val_rows.extend(rows)
    else:
        for i, row in enumerate(rows):
            (train_rows if i % 2 == 0 else val_rows).append(row)

print(f"Split: {len(train_rows)} train  |  {len(val_rows)} val  |  {len(test_rows)} held-out test")


def make_box_file(img_path: Path, gt_text: str) -> str:
    """Generate a Tesseract LSTM .box file content from image dimensions + gt text."""
    from PIL import Image as PILImage
    width, height = PILImage.open(img_path).size
    line = unicodedata.normalize("NFC", gt_text.strip())
    lines = []
    if line:
        for i in range(1, len(line)):
            char = line[i]
            prev_char = line[i - 1]
            if unicodedata.combining(char):
                lines.append(f"{prev_char + char} 0 0 {width} {height} 0")
            elif not unicodedata.combining(prev_char):
                lines.append(f"{prev_char} 0 0 {width} {height} 0")
        if not unicodedata.combining(line[-1]):
            lines.append(f"{line[-1]} 0 0 {width} {height} 0")
        lines.append(f"\t 0 0 {width} {height} 0")
    return "\n".join(lines) + "\n"


def generate_lstmf(row: dict, dest_dir: Path) -> Path | None:
    img_src = TRAINING_DIR / row["image_file"]
    gt_src = TRAINING_DIR / row["ground_truth_file"]
    if not img_src.is_file() or not gt_src.is_file():
        print(f"  SKIP (missing): {img_src}")
        return None

    base = img_src.stem
    dest_img = dest_dir / img_src.name
    shutil.copy2(img_src, dest_img)

    gt_text = gt_src.read_text(encoding="utf-8").strip()

    # Write .box file (character-level bounding boxes for the whole image)
    box_path = dest_dir / f"{base}.box"
    box_path.write_text(make_box_file(dest_img, gt_text), encoding="utf-8")

    # tesseract reads: image + .box → produces .lstmf
    # --tessdata-dir must point to the float model for the lstm unicharset
    result = subprocess.run(
        [
            "tesseract",
            str(dest_img),
            str(dest_dir / base),
            "--tessdata-dir", str(TESSDATA_BEST),
            "-l", "eng",
            str(SYSTEM_TESSDATA / "configs" / "lstm.train"),
        ],
        capture_output=True, text=True,
        env={
            **__import__("os").environ,
            "TESSDATA_PREFIX": str(TESSDATA_BEST),
        }
    )
    lstmf_path = dest_dir / f"{base}.lstmf"
    if not lstmf_path.is_file():
        print(f"  FAIL: {base}")
        print(f"    stdout: {result.stdout.strip()[:200]}")
        print(f"    stderr: {result.stderr.strip()[:200]}")
        return None
    return lstmf_path


print("\n── Generating .lstmf for TRAIN ──")
train_lstmf = []
for row in train_rows:
    p = generate_lstmf(row, TRAIN_DIR)
    if p:
        train_lstmf.append(str(p))
        print(f"  OK  {p.name}")

print("\n── Generating .lstmf for VAL ──")
val_lstmf = []
for row in val_rows:
    p = generate_lstmf(row, VAL_DIR)
    if p:
        val_lstmf.append(str(p))
        print(f"  OK  {p.name}")

(TRAINING_DIR / "train_list.txt").write_text("\n".join(train_lstmf) + "\n", encoding="utf-8")
(TRAINING_DIR / "val_list.txt").write_text("\n".join(val_lstmf) + "\n", encoding="utf-8")

print(f"\nDone: {len(train_lstmf)} train  |  {len(val_lstmf)} val")
if len(train_lstmf) < 3:
    print("ERROR: Too few training .lstmf files. Cannot proceed.")
    sys.exit(1)
