
import unittest

import torch

from src.model.language_model import CoastMindLanguageModel


class TestCoastMindLanguageModel(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = CoastMindLanguageModel(
            vocab_size=32,
            embedding_dim=8,
            num_heads=4,
            hidden_dim=32,
            num_layers=2,
            max_seq_length=16,
        )
        self.token_ids = torch.randint(0, 32, (2, 4))

    def test_logits_have_expected_shape(self):
        logits = self.model(self.token_ids)

        self.assertEqual(logits.shape, (2, 4, 32))

    def test_output_is_a_tensor(self):
        logits = self.model(self.token_ids)

        self.assertIsInstance(logits, torch.Tensor)

    def test_logits_are_finite(self):
        logits = self.model(self.token_ids)

        self.assertTrue(torch.isfinite(logits).all().item())

    def test_model_has_expected_number_of_blocks(self):
        self.assertEqual(len(self.model.transformer_blocks), 2)

    def test_gradients_flow_through_model(self):
        logits = self.model(self.token_ids)
        loss = logits.square().mean()
        loss.backward()

        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(
                    torch.isfinite(parameter.grad).all().item()
                )

    def test_rejects_zero_vocabulary_size(self):
        with self.assertRaises(ValueError):
            CoastMindLanguageModel(
                vocab_size=0,
                embedding_dim=8,
                num_heads=4,
                hidden_dim=32,
                num_layers=2,
                max_seq_length=16,
            )

    def test_rejects_zero_number_of_layers(self):
        with self.assertRaises(ValueError):
            CoastMindLanguageModel(
                vocab_size=32,
                embedding_dim=8,
                num_heads=4,
                hidden_dim=32,
                num_layers=0,
                max_seq_length=16,
            )


if __name__ == "__main__":
    unittest.main()
