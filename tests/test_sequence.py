
import unittest

import torch

from src.training.sequence import create_input_target_pairs


class TestCreateInputTargetPairs(unittest.TestCase):
    def test_creates_expected_input_and_target(self):
        x, y = create_input_target_pairs([4, 7, 2, 9, 5], 4)

        self.assertEqual(x.tolist(), [[4, 7, 2, 9]])
        self.assertEqual(y.tolist(), [[7, 2, 9, 5]])

    def test_creates_sliding_windows(self):
        x, y = create_input_target_pairs([1, 2, 3, 4, 5, 6], 3)

        self.assertEqual(
            x.tolist(),
            [
                [1, 2, 3],
                [2, 3, 4],
                [3, 4, 5],
            ],
        )
        self.assertEqual(
            y.tolist(),
            [
                [2, 3, 4],
                [3, 4, 5],
                [4, 5, 6],
            ],
        )

    def test_returns_long_tensors(self):
        x, y = create_input_target_pairs([1, 2, 3, 4], 2)

        self.assertIsInstance(x, torch.Tensor)
        self.assertIsInstance(y, torch.Tensor)
        self.assertEqual(x.dtype, torch.long)
        self.assertEqual(y.dtype, torch.long)

    def test_rejects_nonpositive_block_size(self):
        with self.assertRaisesRegex(ValueError, "greater than zero"):
            create_input_target_pairs([1, 2, 3], 0)

    def test_rejects_sequence_not_long_enough(self):
        with self.assertRaisesRegex(ValueError, "more tokens than block_size"):
            create_input_target_pairs([1, 2, 3], 3)


if __name__ == "__main__":
    unittest.main()
