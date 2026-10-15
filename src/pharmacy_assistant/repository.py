"""Read-only catalog and inventory queries."""

import sqlite3

from pharmacy_assistant.models import InventoryEntry, Medicine, StorageLocation


def medicine_from_row(row: sqlite3.Row) -> Medicine:
    return Medicine(
        id=row["id"],
        generic_name=row["generic_name"],
        brand_name=row["brand_name"],
        strength=row["strength"],
        dosage_form=row["dosage_form"],
    )


def location_from_row(row: sqlite3.Row) -> StorageLocation:
    return StorageLocation(
        id=row["id"],
        code=row["code"],
        x=row["x"],
        y=row["y"],
        yaw=row["yaw"],
    )


class MedicineRepository:
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection

    def find_by_name(self, normalized_name: str) -> tuple[Medicine, ...]:
        rows = self.connection.execute(
            """
            SELECT DISTINCT m.*
            FROM medicines AS m
            LEFT JOIN medicine_aliases AS a ON a.medicine_id = m.id
            WHERE lower(m.generic_name) = ?
               OR lower(m.brand_name) = ?
               OR lower(a.alias) = ?
            ORDER BY m.id
            """,
            (normalized_name, normalized_name, normalized_name),
        ).fetchall()
        return tuple(medicine_from_row(row) for row in rows)

    def inventory_for_medicine(self, medicine_id: int) -> tuple[InventoryEntry, ...]:
        rows = self.connection.execute(
            """
            SELECT m.*, l.id AS location_id, l.code, l.x, l.y, l.yaw, i.quantity
            FROM inventory AS i
            JOIN medicines AS m ON m.id = i.medicine_id
            JOIN storage_locations AS l ON l.id = i.location_id
            WHERE i.medicine_id = ? AND i.quantity > 0
            ORDER BY i.quantity DESC, l.code
            """,
            (medicine_id,),
        ).fetchall()
        entries: list[InventoryEntry] = []
        for row in rows:
            entries.append(
                InventoryEntry(
                    medicine=medicine_from_row(row),
                    location=StorageLocation(
                        id=row["location_id"],
                        code=row["code"],
                        x=row["x"],
                        y=row["y"],
                        yaw=row["yaw"],
                    ),
                    quantity=row["quantity"],
                )
            )
        return tuple(entries)

