
import json
import tempfile
import unittest
from pathlib import Path

from src.data.dataset import load_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestLoadCorpus(unittest.TestCase):
    def test_loads_pilot_corpus(self):
        path = PROJECT_ROOT / "data" / "raw" / "pilot_corpus.jsonl"

        records = load_corpus(path)

        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["record_id"], "SYN-001")

    def test_rejects_malformed_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "broken.jsonl"
            path.write_text("{invalid json}\n", encoding="utf-8")

            with self.assertRaises(ValueError):
                load_corpus(path)

    def test_rejects_missing_required_field(self):
        record = {
            "record_id": "TEST-001",
            "text": "A coastal science example.",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "missing.jsonl"
            path.write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing fields"):
                load_corpus(path)

    def test_rejects_duplicate_record_ids(self):
        record = {
            "record_id": "TEST-001",
            "text": "A coastal science example.",
            "source_type": "synthetic",
            "source_title": None,
            "source_url": None,
            "license": "Original synthetic educational text",
            "topic": "coastal_processes",
            "collection_date": "2026-10-09",
            "quality_status": "synthetic_example",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "duplicates.jsonl"
            path.write_text(
                json.dumps(record) + "\n" + json.dumps(record) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Duplicate record_id"):
                load_corpus(path)

    def test_rejects_empty_text(self):
        record = {
            "record_id": "TEST-002",
            "text": "   ",
            "source_type": "synthetic",
            "source_title": None,
            "source_url": None,
            "license": "Original synthetic educational text",
            "topic": "coastal_processes",
            "collection_date": "2026-10-09",
            "quality_status": "synthetic_example",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "empty_text.jsonl"
            path.write_text(
                json.dumps(record) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "empty or invalid text"):
                load_corpus(path)


if __name__ == "__main__":
    unittest.main()
