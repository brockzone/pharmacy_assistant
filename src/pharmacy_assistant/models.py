"""Typed domain models used by the catalog and retrieval workflow."""

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Medicine:
    id: int
    generic_name: str
    brand_name: str | None
    strength: str
    dosage_form: str


@dataclass(frozen=True)
class StorageLocation:
    id: int
    code: str
    x: float
    y: float
    yaw: float


@dataclass(frozen=True)
class InventoryEntry:
    medicine: Medicine
    location: StorageLocation
    quantity: int


class MatchStatus(str, Enum):
    MATCHED = "matched"
    AMBIGUOUS = "ambiguous"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class MatchResult:
    status: MatchStatus
    candidates: tuple[Medicine, ...]

    @property
    def medicine(self) -> Medicine | None:
        if self.status is MatchStatus.MATCHED:
            return self.candidates[0]
        return None


class RetrievalItemStatus(str, Enum):
    READY = "ready"
    NEEDS_REVIEW = "needs_review"
    NOT_FOUND = "not_found"
    OUT_OF_STOCK = "out_of_stock"


@dataclass(frozen=True)
class RetrievalItem:
    requested_name: str
    requested_strength: str | None
    requested_dosage_form: str | None
    status: RetrievalItemStatus
    medicine: Medicine | None = None
    location: StorageLocation | None = None
    quantity_available: int = 0
    candidate_medicine_ids: tuple[int, ...] = ()
    message: str | None = None


@dataclass(frozen=True)
class RetrievalRequest:
    request_id: str
    items: tuple[RetrievalItem, ...]

