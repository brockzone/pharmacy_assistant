"""Automated tests for the Phase 1 catalog and retrieval workflow."""

import sqlite3
import tempfile
import unittest
from pathlib import Path

from pharmacy_assistant.database import connect, initialize_database, seed_demo_data
from pharmacy_assistant.matching import MedicineMatcher
from pharmacy_assistant.models import MatchStatus, RetrievalItemStatus
from pharmacy_assistant.resolver import RetrievalRequestBuilder


class PhaseOneTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temp_directory.name) / "test.db"
        initialize_database(self.database_path)
        seed_demo_data(self.database_path)
        self.connection = connect(self.database_path)

    def tearDown(self) -> None:
        self.connection.close()
        self.temp_directory.cleanup()

    def test_schema_initializes_and_enforces_nonnegative_inventory(self) -> None:
        with self.assertRaises(sqlite3.IntegrityError):
            self.connection.execute(
                "INSERT INTO inventory VALUES (1, 1, -1)"
            )

    def test_demo_seeding_is_idempotent_with_unrelated_medicines(self) -> None:
        self.connection.execute(
            """
            INSERT INTO medicines (generic_name, brand_name, strength, dosage_form)
            VALUES ('unrelated demo test', NULL, '1 mg', 'tablet')
            """
        )
        self.connection.commit()
        seed_demo_data(self.database_path)
        count = self.connection.execute(
            "SELECT COUNT(*) FROM medicines WHERE generic_name = 'unrelated demo test'"
        ).fetchone()[0]
        self.assertEqual(count, 1)

    def test_matches_generic_brand_alias_strength_and_form(self) -> None:
        matcher = MedicineMatcher(self.connection)
        self.assertEqual(
            matcher.match(" PARACETAMOL ", "500 MG", "TABLET").status,
            MatchStatus.MATCHED,
        )
        self.assertEqual(
            matcher.match("Fictional Relief", "500 mg", "tablet").medicine.id,
            1,
        )
        self.assertEqual(
            matcher.match("acetaminophen", "500 mg").medicine.id,
            1,
        )

    def test_missing_details_that_leave_multiple_matches_require_review(self) -> None:
        result = MedicineMatcher(self.connection).match("paracetamol")
        self.assertEqual(result.status, MatchStatus.AMBIGUOUS)
        self.assertEqual({medicine.id for medicine in result.candidates}, {1, 2})

    def test_mismatched_strength_is_not_substituted(self) -> None:
        result = MedicineMatcher(self.connection).match("paracetamol", "250 mg")
        self.assertEqual(result.status, MatchStatus.NOT_FOUND)
        request = RetrievalRequestBuilder(self.connection).build(
            [{"name": "paracetamol", "strength": "250 mg"}]
        )
        self.assertEqual(request.items[0].status, RetrievalItemStatus.NOT_FOUND)

    def test_retrieval_request_resolves_fictional_location(self) -> None:
        request = RetrievalRequestBuilder(self.connection).build(
            [{"name": "paracetamol", "strength": "500 mg", "dosage_form": "tablet"}]
        )
        item = request.items[0]
        self.assertEqual(item.status, RetrievalItemStatus.READY)
        self.assertEqual(item.location.code, "DEMO-A1")
        self.assertEqual(item.quantity_available, 12)
        self.assertTrue(request.request_id)

    def test_ambiguous_request_is_not_dispatch_ready(self) -> None:
        request = RetrievalRequestBuilder(self.connection).build(
            [{"name": "paracetamol"}]
        )
        self.assertEqual(request.items[0].status, RetrievalItemStatus.NEEDS_REVIEW)
        self.assertEqual(request.items[0].candidate_medicine_ids, (1, 2))


if __name__ == "__main__":
    unittest.main()
