
import unittest

import torch

from src.model.transformer_block import FeedForward, TransformerBlock


class TestFeedForward(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.embedding_dim = 8
        self.hidden_dim = 32
        self.model = FeedForward(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
        )
        self.x = torch.randn(2, 4, self.embedding_dim)

    def test_output_shape_matches_input_shape(self):
        output = self.model(self.x)

        self.assertEqual(output.shape, self.x.shape)

    def test_output_is_a_tensor(self):
        output = self.model(self.x)

        self.assertIsInstance(output, torch.Tensor)

    def test_outputs_are_finite(self):
        output = self.model(self.x)

        self.assertTrue(torch.isfinite(output).all().item())

    def test_parameters_are_trainable(self):
        for parameter in self.model.parameters():
            self.assertTrue(parameter.requires_grad)

    def test_gradients_flow_through_model(self):
        output = self.model(self.x)
        loss = output.square().mean()
        loss.backward()

        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(
                    torch.isfinite(parameter.grad).all().item()
                )

    def test_rejects_invalid_embedding_dimension(self):
        with self.assertRaises(ValueError):
            FeedForward(embedding_dim=0, hidden_dim=32)

    def test_rejects_invalid_hidden_dimension(self):
        with self.assertRaises(ValueError):
            FeedForward(embedding_dim=8, hidden_dim=0)

class TestTransformerBlock(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.model = TransformerBlock(
            embedding_dim=8,
            num_heads=4,
            hidden_dim=32,
        )
        self.x = torch.randn(2, 4, 8)

    def test_output_shape_matches_input_shape(self):
        output = self.model(self.x)
        self.assertEqual(output.shape, self.x.shape)

    def test_output_is_a_tensor(self):
        output = self.model(self.x)
        self.assertIsInstance(output, torch.Tensor)

    def test_outputs_are_finite(self):
        output = self.model(self.x)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_parameters_are_trainable(self):
        for parameter in self.model.parameters():
            self.assertTrue(parameter.requires_grad)

    def test_gradients_flow_through_model(self):
        output = self.model(self.x)
        loss = output.square().mean()
        loss.backward()

        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(
                    torch.isfinite(parameter.grad).all().item()
                )

if __name__ == "__main__":
    unittest.main()
