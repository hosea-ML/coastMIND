
import tempfile
import unittest
from pathlib import Path

import torch

from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.training.prepare_training_data import prepare_training_data
from src.training.trainer import create_model


class TestCheckpoint(unittest.TestCase):
    def setUp(self):
        self.training_data = prepare_training_data(block_size=32)
        self.tokenizer = self.training_data["tokenizer"]
        self.model = create_model(len(self.tokenizer.itos))
        self.model.eval()

        self.temp_directory = tempfile.TemporaryDirectory()
        self.checkpoint_path = (
            Path(self.temp_directory.name) / "coastmind_checkpoint.pt"
        )

    def tearDown(self):
        self.temp_directory.cleanup()

    def test_saves_checkpoint_file(self):
        save_checkpoint(
            self.model,
            self.tokenizer,
            self.checkpoint_path,
        )

        self.assertTrue(self.checkpoint_path.exists())

    def test_loads_model_and_tokenizer(self):
        save_checkpoint(
            self.model,
            self.tokenizer,
            self.checkpoint_path,
        )

        loaded_model, loaded_tokenizer = load_checkpoint(
            self.checkpoint_path
        )

        self.assertEqual(
            loaded_model.vocab_size,
            self.model.vocab_size,
        )
        self.assertEqual(
            loaded_tokenizer.vocab,
            self.tokenizer.vocab,
        )
        self.assertFalse(loaded_model.training)

    def test_loaded_model_produces_same_logits(self):
        save_checkpoint(
            self.model,
            self.tokenizer,
            self.checkpoint_path,
        )

        loaded_model, _ = load_checkpoint(self.checkpoint_path)
        inputs = self.training_data["inputs"][:2]

        with torch.no_grad():
            original_logits = self.model(inputs)
            loaded_logits = loaded_model(inputs)

        torch.testing.assert_close(
            original_logits,
            loaded_logits,
        )

    def test_preserves_tokenizer_round_trip(self):
        save_checkpoint(
            self.model,
            self.tokenizer,
            self.checkpoint_path,
        )

        _, loaded_tokenizer = load_checkpoint(self.checkpoint_path)
        sample_text = self.training_data["records"][0]["text"]

        encoded = loaded_tokenizer.encode(sample_text)
        decoded = loaded_tokenizer.decode(encoded)

        self.assertEqual(decoded, sample_text)

    def test_rejects_unsupported_checkpoint_version(self):
        self.checkpoint_path.write_bytes(b"not a valid checkpoint")

        with self.assertRaises(Exception):
            load_checkpoint(self.checkpoint_path)


if __name__ == "__main__":
    unittest.main()
