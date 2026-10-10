
import unittest

import torch

from src.model.attention import (
    QKVProjection,
    AttentionScores,
    AttentionScaling,
    AttentionSoftmax,
    CausalMask,
    ValueAggregation,
    SelfAttention,
)


class TestQKVProjection(unittest.TestCase):

    def setUp(self):
        self.embedding_dim = 16
        self.projection = QKVProjection(
            embedding_dim=self.embedding_dim
        )

    def test_output_shapes(self):
        x = torch.randn(2, 4, self.embedding_dim)

        queries, keys, values = self.projection(x)

        expected_shape = (2, 4, self.embedding_dim)

        self.assertEqual(queries.shape, expected_shape)
        self.assertEqual(keys.shape, expected_shape)
        self.assertEqual(values.shape, expected_shape)

    def test_outputs_are_tensors(self):
        x = torch.randn(2, 4, self.embedding_dim)

        outputs = self.projection(x)

        for output in outputs:
            self.assertIsInstance(output, torch.Tensor)

    def test_projection_parameters_are_trainable(self):
        self.assertTrue(self.projection.query.weight.requires_grad)
        self.assertTrue(self.projection.key.weight.requires_grad)
        self.assertTrue(self.projection.value.weight.requires_grad)

    def test_three_projections_have_separate_weights(self):
        self.assertIsNot(
            self.projection.query.weight,
            self.projection.key.weight,
        )
        self.assertIsNot(
            self.projection.query.weight,
            self.projection.value.weight,
        )
        self.assertIsNot(
            self.projection.key.weight,
            self.projection.value.weight,
        )


class TestAttentionScores(unittest.TestCase):

    def setUp(self):
        self.attention_scores = AttentionScores()

    def test_output_shape(self):
        queries = torch.randn(2, 4, 16)
        keys = torch.randn(2, 4, 16)

        scores = self.attention_scores(queries, keys)

        self.assertEqual(scores.shape, (2, 4, 4))

    def test_output_is_a_tensor(self):
        queries = torch.randn(1, 3, 8)
        keys = torch.randn(1, 3, 8)

        scores = self.attention_scores(queries, keys)

        self.assertIsInstance(scores, torch.Tensor)

    def test_scores_match_manual_calculation(self):
        queries = torch.tensor([[[1.0, 2.0]]])
        keys = torch.tensor([[[3.0, 4.0]]])

        scores = self.attention_scores(queries, keys)

        expected = torch.tensor([[[11.0]]])

        self.assertTrue(torch.allclose(scores, expected))

    def test_each_query_is_compared_with_each_key(self):
        queries = torch.randn(2, 4, 16)
        keys = torch.randn(2, 4, 16)

        scores = self.attention_scores(queries, keys)

        self.assertEqual(scores.shape[1], queries.shape[1])
        self.assertEqual(scores.shape[2], keys.shape[1])

class TestAttentionScaling(unittest.TestCase):

    def setUp(self):
        self.scaling = AttentionScaling(key_dim=16)

    def test_scales_scores_correctly(self):
        scores = torch.tensor([[12.0, 8.0], [4.0, 16.0]])

        scaled_scores = self.scaling(scores)

        expected = torch.tensor([[3.0, 2.0], [1.0, 4.0]])

        self.assertTrue(torch.allclose(scaled_scores, expected))

    def test_preserves_output_shape(self):
        scores = torch.randn(2, 4, 4)

        scaled_scores = self.scaling(scores)

        self.assertEqual(scaled_scores.shape, scores.shape)

    def test_rejects_zero_key_dimension(self):
        with self.assertRaises(ValueError):
            AttentionScaling(key_dim=0)

    def test_rejects_negative_key_dimension(self):
        with self.assertRaises(ValueError):
            AttentionScaling(key_dim=-1)

class TestAttentionSoftmax(unittest.TestCase):

    def setUp(self):
        self.softmax = AttentionSoftmax()

    def test_weights_sum_to_one(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])

        weights = self.softmax(scores)

        row_sums = weights.sum(dim=-1)

        self.assertTrue(
            torch.allclose(row_sums, torch.ones_like(row_sums))
        )

    def test_preserves_output_shape(self):
        scores = torch.randn(2, 4, 4)

        weights = self.softmax(scores)

        self.assertEqual(weights.shape, scores.shape)

    def test_weights_are_nonnegative(self):
        scores = torch.randn(2, 4, 4)

        weights = self.softmax(scores)

        self.assertTrue(torch.all(weights >= 0).item())

    def test_matches_manual_example(self):
        scores = torch.tensor([[1.0, 2.0, 3.0]])

        weights = self.softmax(scores)

        expected = torch.tensor([[0.09003057, 0.24472848, 0.66524094]])

        self.assertTrue(torch.allclose(weights, expected, atol=1e-6))

class TestCausalMask(unittest.TestCase):
    def setUp(self):
        self.causal_mask = CausalMask()

    def test_mask_has_correct_shape(self):
        mask = self.causal_mask(4)

        self.assertEqual(mask.shape, (4, 4))

    def test_mask_uses_boolean_values(self):
        mask = self.causal_mask(4)

        self.assertEqual(mask.dtype, torch.bool)

    def test_diagonal_and_past_positions_are_allowed(self):
        mask = self.causal_mask(4)

        expected = torch.tensor([
            [True,  False, False, False],
            [True,  True,  False, False],
            [True,  True,  True,  False],
            [True,  True,  True,  True],
        ])

        self.assertTrue(torch.equal(mask, expected))

    def test_future_positions_are_blocked(self):
        mask = self.causal_mask(4)

        self.assertFalse(mask[0, 1].item())
        self.assertFalse(mask[0, 3].item())
        self.assertFalse(mask[1, 2].item())
        self.assertFalse(mask[2, 3].item())

    def test_zero_sequence_length_raises_error(self):
        with self.assertRaises(ValueError):
            self.causal_mask(0)

    def test_negative_sequence_length_raises_error(self):
        with self.assertRaises(ValueError):
            self.causal_mask(-1)

    
    def test_future_attention_weights_are_zero_after_softmax(self):
        scores = torch.tensor([
            [2.0, 1.5, 3.0, 0.5],
            [1.0, 2.0, 1.5, 0.5],
            [0.5, 1.0, 2.0, 1.5],
            [1.0, 0.5, 1.5, 2.0],
        ])

        mask = self.causal_mask(4)

        masked_scores = scores.masked_fill(~mask, float("-inf"))
        attention_weights = torch.softmax(masked_scores, dim=-1)

        self.assertTrue(
            torch.all(attention_weights[~mask] == 0).item()
        )

        self.assertTrue(
            torch.allclose(
                attention_weights.sum(dim=-1),
                torch.ones(4),
            )
        )


class TestValueAggregation(unittest.TestCase):
    def setUp(self):
        self.aggregation = ValueAggregation()

    def test_output_shape(self):
        attention_weights = torch.rand(2, 4, 4)
        values = torch.rand(2, 4, 8)

        context = self.aggregation(attention_weights, values)

        self.assertEqual(context.shape, (2, 4, 8))

    def test_matches_manual_calculation(self):
        attention_weights = torch.tensor([
            [0.25, 0.75]
        ])

        values = torch.tensor([
            [2.0, 4.0],
            [6.0, 8.0],
        ])

        context = self.aggregation(attention_weights, values)

        expected = torch.tensor([
            [5.0, 7.0]
        ])

        self.assertTrue(torch.allclose(context, expected))

    def test_single_token_preserves_its_value(self):
        attention_weights = torch.tensor([
            [1.0]
        ])

        values = torch.tensor([
            [3.0, 5.0, 7.0]
        ])

        context = self.aggregation(attention_weights, values)

        self.assertTrue(torch.equal(context, values))

    def test_zero_weights_produce_zero_context(self):
        attention_weights = torch.zeros(2, 3, 3)
        values = torch.rand(2, 3, 4)

        context = self.aggregation(attention_weights, values)

        self.assertTrue(torch.equal(context, torch.zeros_like(context)))

class TestSelfAttention(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.embedding_dim = 8
        self.self_attention = SelfAttention(self.embedding_dim)

    def test_output_shape(self):
        x = torch.randn(2, 4, self.embedding_dim)

        output = self.self_attention(x)

        self.assertEqual(output.shape, (2, 4, self.embedding_dim))

    def test_output_is_a_tensor_by_default(self):
        x = torch.randn(2, 4, self.embedding_dim)

        output = self.self_attention(x)

        self.assertIsInstance(output, torch.Tensor)

    def test_attention_weights_sum_to_one(self):
        x = torch.randn(2, 4, self.embedding_dim)

        _, attention_weights = self.self_attention(
            x,
            return_attention=True,
        )

        expected = torch.ones(2, 4)

        self.assertTrue(
            torch.allclose(
                attention_weights.sum(dim=-1),
                expected,
            )
        )

    def test_future_attention_weights_are_zero(self):
        x = torch.randn(2, 4, self.embedding_dim)

        _, attention_weights = self.self_attention(
            x,
            return_attention=True,
        )

        future_positions = torch.triu(
            torch.ones(4, 4, dtype=torch.bool),
            diagonal=1,
        )

        self.assertTrue(
            torch.all(attention_weights[:, future_positions] == 0).item()
        )

    def test_rejects_nonpositive_embedding_dimension(self):
        for embedding_dim in (0, -1):
            with self.subTest(embedding_dim=embedding_dim):
                with self.assertRaises(ValueError):
                    SelfAttention(embedding_dim)

if __name__ == "__main__":
    unittest.main()
