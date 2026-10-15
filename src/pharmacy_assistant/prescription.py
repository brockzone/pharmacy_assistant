"""Prescription image OCR and conservative medicine-candidate matching."""

from dataclasses import dataclass
from pathlib import Path
import re
import unicodedata

import cv2
import pytesseract

from pharmacy_assistant.matching import MedicineMatcher
from pharmacy_assistant.models import MatchStatus, Medicine, StorageLocation
from pharmacy_assistant.repository import MedicineRepository

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

_STRENGTH = re.compile(
    r"(?<![\w.])\d+(?:[.,]\d+)?\s*(?:mcg|μg|ug|mg|g|ml|l|%)"
    r"(?:\s*/\s*\d+(?:[.,]\d+)?\s*(?:mcg|μg|ug|mg|g|ml|l))?(?![\w])",
    re.IGNORECASE,
)
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_FORM_PREFIX = re.compile(
    r"^\s*(?:tab(?:let)?\.?|cap(?:sule)?\.?|syrup|suspension|"
    r"injection|inj\.?|solution|sachet)\s*[:.-]?\s+",
    re.IGNORECASE,
)
_FORM_TERM = re.compile(
    r"\b(oral\s+suspension|tablet|tab\.?|capsule|cap\.?|syrup|"
    r"suspension|injection|inj\.?|solution|sachet)\b",
    re.IGNORECASE,
)
_CANONICAL_FORMS = {
    "tab": "tablet",
    "tab.": "tablet",
    "cap": "capsule",
    "cap.": "capsule",
    "inj": "injection",
    "inj.": "injection",
}
_NON_MEDICINE_PREFIX = re.compile(
    r"^(?:take|give|use|apply|patient|doctor|date|age|diagnosis|"
    r"signature|quantity|qty|refill|address|directions?)\b",
    re.IGNORECASE,
)
_DOSAGE_SCHEDULE = re.compile(
    r"^\s*[-–—:;,]?\s*\d+(?:\s*[-/]\s*\d+){1,}(?:\s|$).*$"
)
_OCR_TOKEN = re.compile(r"[^\W\d_]+(?:['-][^\W\d_]+)*", re.UNICODE)
_LIKELY_MEDICINE_WITHOUT_STRENGTH = re.compile(
    r"^\s*(?:tab(?:let)?\.?|cap(?:sule)?\.?|syrup|suspension|"
    r"injection|inj\.?|solution|sachet)\s+(.+?)\s*$",
    re.IGNORECASE,
)


class PrescriptionError(Exception):
    """Base class for image, camera, and OCR failures."""


class PrescriptionImageError(PrescriptionError):
    """The supplied image path or image data cannot be processed."""


class PrescriptionOCRError(PrescriptionError):
    """Tesseract failed while recognizing prescription text."""


class PrescriptionDebugError(PrescriptionError):
    """OCR debug artifacts could not be saved."""


class TesseractUnavailableError(PrescriptionOCRError):
    """The Tesseract executable is not installed or is not on PATH."""


class CameraCaptureError(PrescriptionError):
    """A webcam could not be opened or did not provide an image."""


class CameraCaptureCancelled(CameraCaptureError):
    """The operator cancelled webcam capture."""


@dataclass(frozen=True)
class PrescriptionCandidate:
    name: str | None
    strength: str | None
    source_line: str
    dosage_form: str | None = None
    review_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class ParsedPrescription:
    candidates: tuple[PrescriptionCandidate, ...]
    unparsed_lines: tuple[str, ...]


@dataclass(frozen=True)
class ScannedMedicine:
    candidate: PrescriptionCandidate
    database_status: str
    medicine: Medicine | None = None
    location: StorageLocation | None = None
    quantity_available: int | None = None
    review_fields: tuple[str, ...] = ()


@dataclass(frozen=True)
class PrescriptionScanResult:
    raw_text: str
    items: tuple[ScannedMedicine, ...]
    unparsed_lines: tuple[str, ...]
    ocr_configuration: str | None = None
    ocr_score: float | None = None


@dataclass(frozen=True)
class OCRAttempt:
    text: str
    configuration: str
    score: float
    image: object


def load_valid_image(image_path: str | Path):
    """Load an allowed image and reject missing, unsupported, or unreadable files."""
    path = Path(image_path).expanduser()
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise PrescriptionImageError("Image must be a JPG, JPEG, or PNG file.")
    try:
        is_file = path.is_file()
    except OSError as error:
        raise PrescriptionImageError(f"Could not access image file: {path}") from error
    if not is_file:
        raise PrescriptionImageError(f"Image file does not exist or is not a file: {path}")
    try:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    except cv2.error as error:
        raise PrescriptionImageError(f"Could not read image file: {path}") from error
    if image is None or image.size == 0:
        raise PrescriptionImageError(f"Image file is unreadable or empty: {path}")
    return image


def preprocess_image(image):
    """Create complementary grayscale, Otsu, and adaptive OCR images."""
    if image is None or getattr(image, "size", 0) == 0:
        raise PrescriptionImageError("No image data was provided.")
    try:
        if image.ndim == 2:
            grayscale = image
        elif image.ndim == 3 and image.shape[2] == 3:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            grayscale = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise PrescriptionImageError("Image has an unsupported color layout.")

        min_dimension = min(grayscale.shape[:2])
        if min_dimension < 1000:
            scale = min(3.0, 1000 / min_dimension)
            grayscale = cv2.resize(
                grayscale,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        original_gray = grayscale.copy()
        denoised = cv2.fastNlMeansDenoising(
            grayscale, None, h=5, templateWindowSize=7, searchWindowSize=21
        )
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
        _, otsu = cv2.threshold(
            enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
        return {
            "original": original_gray,
            "grayscale": enhanced,
            "otsu": otsu,
            "adaptive": adaptive,
        }
    except cv2.error as error:
        raise PrescriptionImageError("Could not preprocess the prescription image.") from error


def _clean_ocr_name(text: str) -> str | None:
    cleaned = _LIST_MARKER.sub("", text, count=1)
    cleaned = re.sub(r"^rx\b\s*[:.-]?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = re.sub(r"[^\w\s'-]", " ", cleaned, flags=re.UNICODE)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_'")
    cleaned = _FORM_PREFIX.sub("", cleaned, count=1).strip()
    return cleaned or None


def _catalog_name_from_ocr(text: str, matcher: MedicineMatcher | None) -> str | None:
    cleaned = _clean_ocr_name(text)
    if cleaned is None or matcher is None:
        return cleaned
    if matcher.match(cleaned).status is not MatchStatus.NOT_FOUND:
        return cleaned

    tokens = _OCR_TOKEN.findall(cleaned)
    for token_count in range(len(tokens), 0, -1):
        for start in range(len(tokens) - token_count + 1):
            phrase = " ".join(tokens[start : start + token_count])
            compact = "".join(tokens[start : start + token_count])
            for variant in dict.fromkeys((phrase, compact)):
                if matcher.match(variant).status is not MatchStatus.NOT_FOUND:
                    return variant
    return cleaned


def _ocr_text_score(text: str, matcher: MedicineMatcher | None) -> float:
    """Rank OCR outputs by exact catalog evidence and basic text legibility."""
    alphanumeric = sum(character.isalnum() for character in text)
    visible = sum(not character.isspace() for character in text)
    if alphanumeric == 0:
        return 0.0

    score = min(alphanumeric, 60) + 10 * (alphanumeric / max(visible, 1))
    if matcher is None:
        return score

    parsed = parse_prescription_text(text, matcher)
    for candidate in parsed.candidates:
        if candidate.name is None:
            continue
        match = matcher.match(
            candidate.name,
            candidate.strength,
            candidate.dosage_form,
        )
        if match.status is MatchStatus.MATCHED:
            score += 100
        elif match.status is MatchStatus.AMBIGUOUS:
            score += 50
    return score


def _save_ocr_debug(
    image,
    attempts: tuple[OCRAttempt, ...],
    selected: OCRAttempt,
    debug_dir: str | Path | None,
) -> None:
    if debug_dir is None:
        return
    directory = Path(debug_dir).expanduser()
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(directory / "captured.png"), image):
            raise PrescriptionDebugError("Could not save captured.png.")
        if not cv2.imwrite(str(directory / "preprocessed.png"), selected.image):
            raise PrescriptionDebugError("Could not save preprocessed.png.")
        report = "\n".join(
            f"{attempt.configuration}\tscore={attempt.score:.2f}\t"
            f"text={attempt.text!r}"
            for attempt in attempts
        )
        (directory / "ocr-attempts.txt").write_text(report + "\n", encoding="utf-8")
    except OSError as error:
        raise PrescriptionDebugError(
            f"Could not write OCR debug files to {directory}: {error}"
        ) from error
    except cv2.error as error:
        raise PrescriptionDebugError(
            f"Could not write OCR debug images to {directory}: {error}"
        ) from error


def recognize_text_details(
    image,
    matcher: MedicineMatcher | None = None,
    debug_dir: str | Path | None = None,
) -> OCRAttempt:
    """Try a few suitable page modes and thresholding variants, selecting by score."""
    variants = preprocess_image(image)
    recipes = (
        ("original", 7),
        ("grayscale", 7),
        ("otsu", 7),
        ("adaptive", 7),
        ("grayscale", 6),
        ("grayscale", 11),
    )
    attempts: list[OCRAttempt] = []
    errors: list[Exception] = []
    try:
        for variant_name, page_mode in recipes:
            config = f"--oem 3 --psm {page_mode}"
            try:
                text = pytesseract.image_to_string(
                    variants[variant_name], config=config
                )
            except pytesseract.TesseractNotFoundError as error:
                raise TesseractUnavailableError(
                    "Tesseract is not installed or is not available on PATH."
                ) from error
            except (pytesseract.TesseractError, OSError) as error:
                errors.append(error)
                continue
            attempts.append(
                OCRAttempt(
                    text=text,
                    configuration=f"{variant_name}, {config}",
                    score=_ocr_text_score(text, matcher),
                    image=variants[variant_name],
                )
            )
    except pytesseract.TesseractNotFoundError as error:
        raise TesseractUnavailableError(
            "Tesseract is not installed or is not available on PATH."
        ) from error

    if not attempts:
        detail = str(errors[-1]) if errors else "No OCR configuration completed."
        raise PrescriptionOCRError(f"Tesseract could not read the image: {detail}")

    selected = max(attempts, key=lambda attempt: attempt.score)
    _save_ocr_debug(image, tuple(attempts), selected, debug_dir)
    return selected


def recognize_text(image, matcher: MedicineMatcher | None = None) -> str:
    """Run configured OCR alternatives and return the best-ranked text."""
    return recognize_text_details(image, matcher).text


def _clean_name(text: str) -> str | None:
    cleaned = _LIST_MARKER.sub("", text, count=1).strip()
    cleaned = re.sub(r"^rx\b\s*[:.-]?\s*", "", cleaned, count=1, flags=re.IGNORECASE)
    cleaned = _FORM_PREFIX.sub("", cleaned, count=1)
    cleaned = cleaned.strip(" \t:;,.()[]{}#-")
    return cleaned or None


def _extract_form(text: str) -> tuple[str | None, str]:
    match = _FORM_TERM.search(text)
    if match is None:
        return None, text
    form = re.sub(r"\s+", " ", match.group(1).strip().casefold())
    return _CANONICAL_FORMS.get(form, form), text[: match.start()] + text[match.end() :]


def _format_strength_spacing(strength: str) -> str:
    return re.sub(
        r"(?<=\d)\s*(?=(?:mcg|μg|µg|ug|mg|ml|g|l|%))",
        " ",
        strength.strip(),
        flags=re.IGNORECASE,
    )


def parse_prescription_text(
    raw_text: str,
    matcher: MedicineMatcher | None = None,
) -> ParsedPrescription:
    """Extract explicit name/strength text and retain uncertain lines for review."""
    candidates: list[PrescriptionCandidate] = []
    unparsed: list[str] = []

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        strength_match = _STRENGTH.search(line)
        if strength_match is not None:
            before_strength = line[: strength_match.start()]
            dosage_form, before_strength = _extract_form(before_strength)
            after_strength = line[strength_match.end() :]
            trailing_form, after_strength = _extract_form(after_strength)
            dosage_form = dosage_form or trailing_form
            if _DOSAGE_SCHEDULE.match(after_strength):
                after_strength = ""
            prefix = before_strength + " " + after_strength
            name = _catalog_name_from_ocr(prefix, matcher)
            if name is None:
                name = _clean_name(prefix)
            if name is not None and _NON_MEDICINE_PREFIX.match(name):
                unparsed.append(line)
                continue
            fields = ("name",) if name is None else ()
            candidates.append(
                PrescriptionCandidate(
                    name=name,
                    strength=_format_strength_spacing(strength_match.group(0)),
                    source_line=line,
                    dosage_form=dosage_form,
                    review_fields=fields,
                )
            )
            continue

        likely_medicine = _LIKELY_MEDICINE_WITHOUT_STRENGTH.match(line)
        if likely_medicine is not None:
            dosage_form, without_form = _extract_form(line)
            name = _clean_name(without_form)
            candidates.append(
                PrescriptionCandidate(
                    name=name,
                    strength=None,
                    source_line=line,
                    dosage_form=dosage_form,
                    review_fields=("strength",) if name else ("name", "strength"),
                )
            )
            continue

        dosage_form, without_form = _extract_form(line)
        name = _catalog_name_from_ocr(without_form, matcher)
        if name and matcher is not None and matcher.match(name).status is not MatchStatus.NOT_FOUND:
            candidates.append(
                PrescriptionCandidate(
                    name=name,
                    strength=None,
                    source_line=line,
                    dosage_form=dosage_form,
                    review_fields=("strength",),
                )
            )
            continue

        unparsed.append(line)

    return ParsedPrescription(tuple(candidates), tuple(unparsed))


def _match_candidates(
    parsed: ParsedPrescription,
    matcher: MedicineMatcher,
    repository: MedicineRepository,
) -> tuple[ScannedMedicine, ...]:
    items: list[ScannedMedicine] = []
    for candidate in parsed.candidates:
        if candidate.name is None:
            items.append(
                ScannedMedicine(
                    candidate=candidate,
                    database_status="incomplete_requires_review",
                    review_fields=candidate.review_fields,
                )
            )
            continue

        match = matcher.match(
            candidate.name,
            candidate.strength,
            candidate.dosage_form,
        )
        if candidate.strength is None:
            review_fields = list(candidate.review_fields)
            if match.status is MatchStatus.AMBIGUOUS:
                review_fields.append("medicine_match")
            items.append(
                ScannedMedicine(
                    candidate=candidate,
                    database_status=(
                        "ambiguous_requires_review"
                        if match.status is MatchStatus.AMBIGUOUS
                        else "incomplete_requires_review"
                    ),
                    review_fields=tuple(dict.fromkeys((*review_fields, "strength"))),
                )
            )
            continue
        if match.status is MatchStatus.AMBIGUOUS:
            items.append(
                ScannedMedicine(
                    candidate=candidate,
                    database_status=match.status.value,
                    review_fields=("medicine_match",),
                )
            )
            continue
        if match.status is MatchStatus.NOT_FOUND:
            items.append(
                ScannedMedicine(
                    candidate=candidate,
                    database_status=match.status.value,
                    review_fields=("medicine_match",),
                )
            )
            continue

        medicine = match.candidates[0]
        inventory = repository.inventory_for_medicine(medicine.id)
        if not inventory:
            items.append(
                ScannedMedicine(
                    candidate=candidate,
                    database_status="matched_no_inventory",
                    medicine=medicine,
                    review_fields=("inventory",),
                )
            )
            continue
        items.append(
            ScannedMedicine(
                candidate=candidate,
                database_status=match.status.value,
                medicine=medicine,
                location=inventory[0].location,
                quantity_available=inventory[0].quantity,
            )
        )
    return tuple(items)


def scan_image_array(
    image,
    matcher: MedicineMatcher,
    repository: MedicineRepository,
    debug_dir: str | Path | None = None,
) -> PrescriptionScanResult:
    """OCR an in-memory image and compare candidates with the existing catalog."""
    ocr = recognize_text_details(image, matcher, debug_dir)
    raw_text = ocr.text
    parsed = parse_prescription_text(raw_text, matcher)
    return PrescriptionScanResult(
        raw_text=raw_text,
        items=_match_candidates(parsed, matcher, repository),
        unparsed_lines=parsed.unparsed_lines,
        ocr_configuration=ocr.configuration,
        ocr_score=ocr.score,
    )


def scan_image(
    image_path: str | Path,
    matcher: MedicineMatcher,
    repository: MedicineRepository,
    debug_dir: str | Path | None = None,
) -> PrescriptionScanResult:
    """Validate and scan a JPG, JPEG, or PNG prescription image."""
    return scan_image_array(
        load_valid_image(image_path), matcher, repository, debug_dir
    )


def capture_webcam_image(camera_index: int = 0):
    """Preview a webcam until capture/cancel, releasing camera and windows."""
    window_name = "Pharmacy Assistant - Prescription Camera"
    try:
        camera = cv2.VideoCapture(camera_index)
    except cv2.error as error:
        raise CameraCaptureError(
            f"Could not initialize webcam {camera_index}: {error}"
        ) from error

    try:
        if not camera.isOpened():
            raise CameraCaptureError(f"Could not open webcam {camera_index}.")
        print(
            "Place the paper with the medicine name in view.\n"
            "Keep the paper in frame; use good lighting and avoid shadows or glare.\n"
            "Hold the camera steady against a plain background.\n"
            "Press SPACE to capture the image.\n"
            "Press ESC or Q to cancel."
        )
        while True:
            success, image = camera.read()
            if not success or image is None or image.size == 0:
                raise CameraCaptureError(
                    f"Webcam {camera_index} did not provide a valid frame."
                )
            preview = image.copy()
            cv2.putText(
                preview,
                "SPACE: capture | ESC/Q: cancel",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            guidance = (
                "Keep paper in frame | Good light | Avoid glare | Hold steady"
            )
            cv2.putText(
                preview,
                guidance,
                (12, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window_name, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == 32:
                return image
            if key in (27, ord("q"), ord("Q")):
                raise CameraCaptureCancelled("Webcam scan cancelled.")
    except cv2.error as error:
        raise CameraCaptureError(f"Webcam preview/capture failed: {error}") from error
    finally:
        try:
            camera.release()
        finally:
            cv2.destroyAllWindows()
