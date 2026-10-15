"""Resolve catalog matches into safe, structured retrieval requests."""

import sqlite3
import uuid

from pharmacy_assistant.matching import MedicineMatcher
from pharmacy_assistant.models import (
    InventoryEntry,
    MatchStatus,
    RetrievalItem,
    RetrievalItemStatus,
    RetrievalRequest,
)
from pharmacy_assistant.repository import MedicineRepository


class LocationResolver:
    def __init__(self, connection: sqlite3.Connection):
        self.repository = MedicineRepository(connection)

    def resolve(self, medicine_id: int) -> InventoryEntry | None:
        entries = self.repository.inventory_for_medicine(medicine_id)
        if not entries:
            return None
        return entries[0]


class RetrievalRequestBuilder:
    def __init__(self, connection: sqlite3.Connection):
        self.matcher = MedicineMatcher(connection)
        self.location_resolver = LocationResolver(connection)

    def build(self, requested_medicines: list[dict[str, str | None]]) -> RetrievalRequest:
        items: list[RetrievalItem] = []
        for requested in requested_medicines:
            name = requested.get("name") or ""
            strength = requested.get("strength")
            dosage_form = requested.get("dosage_form")
            match = self.matcher.match(name, strength, dosage_form)

            if match.status is MatchStatus.NOT_FOUND:
                items.append(
                    RetrievalItem(
                        requested_name=name,
                        requested_strength=strength,
                        requested_dosage_form=dosage_form,
                        status=RetrievalItemStatus.NOT_FOUND,
                        message="No exact catalog match; no substitute selected.",
                    )
                )
                continue
            if match.status is MatchStatus.AMBIGUOUS:
                items.append(
                    RetrievalItem(
                        requested_name=name,
                        requested_strength=strength,
                        requested_dosage_form=dosage_form,
                        status=RetrievalItemStatus.NEEDS_REVIEW,
                        candidate_medicine_ids=tuple(
                            medicine.id for medicine in match.candidates
                        ),
                        message="Multiple catalog matches require operator confirmation.",
                    )
                )
                continue

            medicine = match.candidates[0]
            inventory = self.location_resolver.resolve(medicine.id)
            if inventory is None:
                items.append(
                    RetrievalItem(
                        requested_name=name,
                        requested_strength=strength,
                        requested_dosage_form=dosage_form,
                        status=RetrievalItemStatus.OUT_OF_STOCK,
                        medicine=medicine,
                        message="No positive inventory quantity is recorded.",
                    )
                )
                continue
            items.append(
                RetrievalItem(
                    requested_name=name,
                    requested_strength=strength,
                    requested_dosage_form=dosage_form,
                    status=RetrievalItemStatus.READY,
                    medicine=medicine,
                    location=inventory.location,
                    quantity_available=inventory.quantity,
                )
            )

        return RetrievalRequest(
            request_id=str(uuid.uuid4()),
            items=tuple(items),
        )
