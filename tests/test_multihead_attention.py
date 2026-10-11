
import unittest

import torch

from src.model.attention import MultiHeadSelfAttention


class TestMultiHeadSelfAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.embedding_dim = 8
        self.num_heads = 4
        self.model = MultiHeadSelfAttention(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
        )
        self.x = torch.randn(2, 4, self.embedding_dim)

    def test_output_shape_matches_input_shape(self):
        output = self.model(self.x)

        self.assertEqual(output.shape, self.x.shape)

    def test_attention_weights_have_correct_shape(self):
        _, attention = self.model(
            self.x,
            return_attention=True,
        )

        self.assertEqual(attention.shape, (2, 4, 4, 4))

    def test_attention_weights_sum_to_one(self):
        _, attention = self.model(
            self.x,
            return_attention=True,
        )

        expected = torch.ones_like(attention.sum(dim=-1))

        torch.testing.assert_close(
            attention.sum(dim=-1),
            expected,
        )

    def test_future_tokens_receive_zero_attention(self):
        _, attention = self.model(
            self.x,
            return_attention=True,
        )

        sequence_length = self.x.size(1)
        future_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
            ),
            diagonal=1,
        )

        future_weights = attention[..., future_mask]

        torch.testing.assert_close(
            future_weights,
            torch.zeros_like(future_weights),
        )

    def test_first_token_attends_only_to_itself(self):
        _, attention = self.model(
            self.x,
            return_attention=True,
        )

        expected = torch.zeros_like(attention[:, :, 0, :])
        expected[..., 0] = 1.0

        torch.testing.assert_close(
            attention[:, :, 0, :],
            expected,
        )

    def test_outputs_are_finite(self):
        output = self.model(self.x)

        self.assertTrue(torch.isfinite(output).all().item())

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

    def test_invalid_embedding_dimension_raises_error(self):
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(
                embedding_dim=0,
                num_heads=2,
            )

    def test_invalid_number_of_heads_raises_error(self):
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(
                embedding_dim=8,
                num_heads=0,
            )

    def test_non_divisible_dimensions_raise_error(self):
        with self.assertRaises(ValueError):
            MultiHeadSelfAttention(
                embedding_dim=10,
                num_heads=4,
            )


if __name__ == "__main__":
    unittest.main()
