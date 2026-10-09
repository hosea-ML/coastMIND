
import unittest

from src.training.prepare_training_data import prepare_training_data


class TestPrepareTrainingData(unittest.TestCase):

    def test_prepares_expected_number_of_examples(self):
        data = prepare_training_data(block_size=32)

        self.assertEqual(len(data["records"]), 3)
        self.assertEqual(data["inputs"].shape, (460, 32))
        self.assertEqual(data["targets"].shape, (460, 32))

    def test_inputs_and_targets_are_aligned(self):
        data = prepare_training_data(block_size=32)

        inputs = data["inputs"]
        targets = data["targets"]

        self.assertTrue(
            (inputs[:, 1:] == targets[:, :-1]).all().item()
        )

    def test_returns_expected_block_size(self):
        data = prepare_training_data(block_size=32)

        self.assertEqual(data["block_size"], 32)

    def test_rejects_invalid_block_size(self):
        with self.assertRaises(ValueError):
            prepare_training_data(block_size=0)


if __name__ == "__main__":
    unittest.main()
