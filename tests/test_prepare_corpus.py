
import json
import tempfile
import unittest
from pathlib import Path

from src.data.prepare_corpus import prepare_corpus


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PILOT_CORPUS_PATH = (
    PROJECT_ROOT / "data" / "raw" / "pilot_corpus.jsonl"
)


class TestPrepareCorpus(unittest.TestCase):
    def test_prepares_pilot_corpus(self):
        result = prepare_corpus(PILOT_CORPUS_PATH)

        self.assertEqual(len(result["records"]), 3)
        self.assertEqual(len(result["text"]), 492)
        self.assertEqual(len(result["token_ids"]), 492)

    def test_vocabulary_contains_unique_characters(self):
        result = prepare_corpus(PILOT_CORPUS_PATH)

        self.assertEqual(
            len(result["tokenizer"].vocab),
            len(set(result["text"])),
        )

    def test_encoding_and_decoding_preserve_text(self):
        result = prepare_corpus(PILOT_CORPUS_PATH)

        reconstructed = result["tokenizer"].decode(
            result["token_ids"]
        )

        self.assertEqual(reconstructed, result["text"])

    def test_rejects_empty_corpus(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            empty_path = Path(temp_dir) / "empty.jsonl"
            empty_path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "empty corpus"
            ):
                prepare_corpus(empty_path)


if __name__ == "__main__":
    unittest.main()
