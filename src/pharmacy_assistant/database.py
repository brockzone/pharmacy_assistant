"""SQLite schema, connection configuration, and fictional demo data."""

import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS medicines (
    id INTEGER PRIMARY KEY,
    generic_name TEXT NOT NULL,
    brand_name TEXT,
    strength TEXT NOT NULL,
    dosage_form TEXT NOT NULL,
    UNIQUE (generic_name, strength, dosage_form)
);

CREATE TABLE IF NOT EXISTS medicine_aliases (
    alias TEXT NOT NULL COLLATE NOCASE,
    medicine_id INTEGER NOT NULL REFERENCES medicines(id) ON DELETE CASCADE,
    PRIMARY KEY (alias, medicine_id)
);

CREATE TABLE IF NOT EXISTS storage_locations (
    id INTEGER PRIMARY KEY,
    code TEXT NOT NULL UNIQUE,
    x REAL NOT NULL,
    y REAL NOT NULL,
    yaw REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS inventory (
    medicine_id INTEGER NOT NULL REFERENCES medicines(id) ON DELETE CASCADE,
    location_id INTEGER NOT NULL REFERENCES storage_locations(id) ON DELETE CASCADE,
    quantity INTEGER NOT NULL CHECK (quantity >= 0),
    PRIMARY KEY (medicine_id, location_id)
);

CREATE INDEX IF NOT EXISTS idx_medicine_aliases_medicine
    ON medicine_aliases(medicine_id);
CREATE INDEX IF NOT EXISTS idx_inventory_location
    ON inventory(location_id);
"""


def connect(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database with row access and foreign keys enabled."""
    connection = sqlite3.connect(str(database_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(database_path: str | Path) -> None:
    """Create the Phase 1 schema if it does not already exist."""
    path = Path(database_path)
    if str(database_path) != ":memory:":
        path.parent.mkdir(parents=True, exist_ok=True)
    with connect(database_path) as connection:
        connection.executescript(SCHEMA)


def seed_demo_data(database_path: str | Path) -> None:
    """Insert fictional medicines, inventory, and test-only coordinates."""
    initialize_database(database_path)
    medicines = (
        ("paracetamol", "Fictional Relief", "500 mg", "tablet", ("acetaminophen",)),
        ("paracetamol", "Fictional Relief", "120 mg/5 mL", "oral suspension", ()),
        ("amoxicillin", "Fictional Amoxi", "250 mg", "capsule", ()),
    )
    locations = (
        ("DEMO-A1", 1.0, 0.0, 0.0),
        ("DEMO-B2", 2.0, 1.0, 1.57),
        ("DEMO-C3", 3.0, 1.0, 3.14),
    )
    demo_medicine_ids: dict[tuple[str, str], int] = {}

    with connect(database_path) as connection:
        for generic_name, brand_name, strength, dosage_form, aliases in medicines:
            connection.execute(
                """
                INSERT OR IGNORE INTO medicines
                    (generic_name, brand_name, strength, dosage_form)
                VALUES (?, ?, ?, ?)
                """,
                (generic_name, brand_name, strength, dosage_form),
            )
            row = connection.execute(
                """
                SELECT id FROM medicines
                WHERE generic_name = ? AND strength = ? AND dosage_form = ?
                """,
                (generic_name, strength, dosage_form),
            ).fetchone()
            if row is None:
                raise RuntimeError("Demo medicine was not available after insertion")
            demo_medicine_ids[(generic_name, strength)] = row["id"]
            connection.executemany(
                "INSERT OR IGNORE INTO medicine_aliases (alias, medicine_id) VALUES (?, ?)",
                ((alias, row["id"]) for alias in aliases),
            )

        connection.executemany(
            """
            INSERT OR IGNORE INTO storage_locations (code, x, y, yaw)
            VALUES (?, ?, ?, ?)
            """,
            locations,
        )
        location_rows = connection.execute(
            "SELECT id, code FROM storage_locations"
        ).fetchall()
        medicine_location = {
            ("paracetamol", "500 mg"): "DEMO-A1",
            ("paracetamol", "120 mg/5 mL"): "DEMO-B2",
            ("amoxicillin", "250 mg"): "DEMO-C3",
        }
        location_ids = {row["code"]: row["id"] for row in location_rows}
        for medicine_key, code in medicine_location.items():
            connection.execute(
                """
                INSERT OR IGNORE INTO inventory (medicine_id, location_id, quantity)
                VALUES (?, ?, ?)
                """,
                (demo_medicine_ids[medicine_key], location_ids[code], 12),
            )
