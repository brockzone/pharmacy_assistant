"""Strict, deterministic matching of known medicine details."""

import re
import sqlite3
import unicodedata

from pharmacy_assistant.models import MatchResult, MatchStatus, Medicine
from pharmacy_assistant.repository import MedicineRepository


def normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold().strip()
    return re.sub(r"\s+", " ", normalized)


class MedicineMatcher:
    def __init__(self, connection: sqlite3.Connection):
        self.repository = MedicineRepository(connection)

    def match(
        self,
        name: str,
        strength: str | None = None,
        dosage_form: str | None = None,
    ) -> MatchResult:
        normalized_name = normalize_text(name)
        if not normalized_name:
            return MatchResult(MatchStatus.NOT_FOUND, ())

        candidates = self.repository.find_by_name(normalized_name)
        if strength is not None:
            normalized_strength = normalize_text(strength)
            candidates = tuple(
                medicine
                for medicine in candidates
                if normalize_text(medicine.strength) == normalized_strength
            )
        if dosage_form is not None:
            normalized_form = normalize_text(dosage_form)
            candidates = tuple(
                medicine
                for medicine in candidates
                if normalize_text(medicine.dosage_form) == normalized_form
            )

        if not candidates:
            return MatchResult(MatchStatus.NOT_FOUND, ())
        if len(candidates) > 1:
            return MatchResult(MatchStatus.AMBIGUOUS, candidates)
        return MatchResult(MatchStatus.MATCHED, candidates)
