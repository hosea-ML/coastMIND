
import unittest

import torch

import tempfile
from pathlib import Path
from src.training.checkpoint import save_checkpoint, load_checkpoint
from src.generation.generate import generate_text
from src.training.prepare_training_data import prepare_training_data
from src.training.trainer import create_model


class TestGenerateText(unittest.TestCase):
    def setUp(self):
        data = prepare_training_data(block_size=32)
        self.tokenizer = data["tokenizer"]
        self.model = create_model(len(self.tokenizer.itos))

    def test_returns_string(self):
        result = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt="coastal",
            max_new_tokens=5,
        )
        self.assertIsInstance(result, str)

    def test_preserves_prompt(self):
        prompt = "coastal"
        result = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=5,
        )
        self.assertTrue(result.startswith(prompt))

    def test_generates_requested_number_of_tokens(self):
        prompt = "coastal"
        max_new_tokens = 5

        result = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )

        self.assertEqual(
            len(result),
            len(prompt) + max_new_tokens,
        )

    def test_zero_new_tokens_returns_prompt(self):
        prompt = "coastal"
        result = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_new_tokens=0,
        )
        self.assertEqual(result, prompt)

    def test_rejects_empty_prompt(self):
        with self.assertRaises(ValueError):
            generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt="",
            )

    def test_rejects_negative_token_count(self):
        with self.assertRaises(ValueError):
            generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt="coastal",
                max_new_tokens=-1,
            )

    def test_rejects_zero_temperature(self):
        with self.assertRaises(ValueError):
            generate_text(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt="coastal",
                temperature=0,
            )

    def test_model_is_in_evaluation_mode_after_generation(self):
        generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt="coastal",
            max_new_tokens=1,
        )
        self.assertFalse(self.model.training)

    def test_generation_after_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pt"

            # Save the model and tokenizer.
            save_checkpoint(
                model=self.model,
                tokenizer=self.tokenizer,
                path=checkpoint_path,
            )

            # Reload the saved model and tokenizer.
            loaded_model, loaded_tokenizer = load_checkpoint(
                checkpoint_path
            )

            # Generate text using the reloaded checkpoint.
            prompt = "coastal"
            result = generate_text(
                model=loaded_model,
                tokenizer=loaded_tokenizer,
                prompt=prompt,
                max_new_tokens=3,
            )

            # Verify that generation works correctly.
            self.assertTrue(result.startswith(prompt))
            self.assertEqual(len(result), len(prompt) + 3)
            self.assertFalse(loaded_model.training)


if __name__ == "__main__":
    unittest.main()
