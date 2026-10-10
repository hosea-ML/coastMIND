
import unittest

import torch

from src.model.embeddings import (
    TokenEmbedding,
    PositionalEmbedding,
    CoastMindEmbedding,
)

class TestTokenEmbedding(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 32
        self.embedding_dim = 16
        self.embedding = TokenEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
        )

    def test_output_shape(self):
        token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        output = self.embedding(token_ids)

        self.assertEqual(output.shape, (2, 4, 16))

    def test_output_is_a_tensor(self):
        token_ids = torch.tensor([[1, 2, 3]])

        output = self.embedding(token_ids)

        self.assertIsInstance(output, torch.Tensor)

    def test_embedding_parameters_are_trainable(self):
        self.assertTrue(self.embedding.embedding.weight.requires_grad)

    def test_embedding_table_shape(self):
        self.assertEqual(
            self.embedding.embedding.weight.shape,
            (self.vocab_size, self.embedding_dim),
        )

    def test_repeated_token_has_same_embedding(self):
        token_ids = torch.tensor([[5, 5]])

        output = self.embedding(token_ids)

        self.assertTrue(torch.equal(output[0, 0], output[0, 1]))

class TestPositionalEmbedding(unittest.TestCase):

    def setUp(self):
        self.max_seq_length = 32
        self.embedding_dim = 16
        self.embedding = PositionalEmbedding(
            max_seq_length=self.max_seq_length,
            embedding_dim=self.embedding_dim,
        )

    def test_output_shape(self):
        position_ids = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])

        output = self.embedding(position_ids)

        self.assertEqual(output.shape, (2, 4, 16))

    def test_output_is_a_tensor(self):
        position_ids = torch.tensor([[0, 1, 2]])

        output = self.embedding(position_ids)

        self.assertIsInstance(output, torch.Tensor)

    def test_embedding_parameters_are_trainable(self):
        self.assertTrue(self.embedding.embedding.weight.requires_grad)

    def test_embedding_table_shape(self):
        self.assertEqual(
            self.embedding.embedding.weight.shape,
            (self.max_seq_length, self.embedding_dim),
        )

    def test_same_position_has_same_embedding(self):
        position_ids = torch.tensor([[3, 3]])

        output = self.embedding(position_ids)

        self.assertTrue(torch.equal(output[0, 0], output[0, 1]))

    def test_rejects_position_outside_embedding_table(self):
        position_ids = torch.tensor([[self.max_seq_length]])

        with self.assertRaises(IndexError):
            self.embedding(position_ids)


class TestCoastMindEmbedding(unittest.TestCase):

    def setUp(self):
        self.vocab_size = 32
        self.embedding_dim = 16
        self.max_seq_length = 32

        self.embedding = CoastMindEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.embedding_dim,
            max_seq_length=self.max_seq_length,
        )

    def test_output_shape(self):
        token_ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])

        output = self.embedding(token_ids)

        self.assertEqual(output.shape, (2, 4, 16))

    def test_output_is_a_tensor(self):
        token_ids = torch.tensor([[1, 2, 3]])

        output = self.embedding(token_ids)

        self.assertIsInstance(output, torch.Tensor)

    def test_combined_embedding_is_trainable(self):
        self.assertTrue(
            self.embedding.token_embedding.embedding.weight.requires_grad
        )
        self.assertTrue(
            self.embedding.position_embedding.embedding.weight.requires_grad
        )

    def test_rejects_sequence_longer_than_maximum(self):
        token_ids = torch.zeros((1, 33), dtype=torch.long)

        with self.assertRaises(ValueError):
            self.embedding(token_ids)

if __name__ == "__main__":
    unittest.main()

