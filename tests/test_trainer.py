
import unittest

import torch
import tempfile
from torch import nn
from pathlib import Path
from torch.optim import AdamW

from configs import training_config as config
from src.model.language_model import CoastMindLanguageModel
from src.training.prepare_training_data import prepare_training_data
from src.training.trainer import (
    create_data_loader,
    create_model,
    train,
    train_one_epoch,
)


class TestCreateDataLoader(unittest.TestCase):
    def test_creates_expected_number_of_batches(self):
        data = prepare_training_data(config.BLOCK_SIZE)
        loader = create_data_loader(data["inputs"], data["targets"])

        expected_batches = (
            len(data["inputs"]) + config.BATCH_SIZE - 1
        ) // config.BATCH_SIZE

        self.assertEqual(len(loader), expected_batches)

    def test_input_and_target_batch_shapes_match(self):
        data = prepare_training_data(config.BLOCK_SIZE)
        loader = create_data_loader(data["inputs"], data["targets"])

        inputs, targets = next(iter(loader))

        self.assertEqual(inputs.shape, targets.shape)
        self.assertEqual(inputs.shape[1], config.BLOCK_SIZE)

    def test_batch_size_does_not_exceed_configuration(self):
        data = prepare_training_data(config.BLOCK_SIZE)
        loader = create_data_loader(data["inputs"], data["targets"])

        for inputs, targets in loader:
            self.assertLessEqual(inputs.size(0), config.BATCH_SIZE)
            self.assertLessEqual(targets.size(0), config.BATCH_SIZE)


class TestTrainOneEpoch(unittest.TestCase):
    def test_returns_finite_loss(self):
        data = prepare_training_data(config.BLOCK_SIZE)
        vocab_size = len(data["tokenizer"].itos)

        model = create_model(vocab_size)
        loader = create_data_loader(data["inputs"], data["targets"])
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

        loss = train_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            loss_function=nn.CrossEntropyLoss(),
        )

        self.assertIsInstance(loss, float)
        self.assertTrue(torch.isfinite(torch.tensor(loss)).item())

    def test_updates_model_parameters(self):
        data = prepare_training_data(config.BLOCK_SIZE)
        vocab_size = len(data["tokenizer"].itos)

        model = create_model(vocab_size)
        loader = create_data_loader(data["inputs"], data["targets"])
        optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE)

        parameters_before = [
            parameter.detach().clone()
            for parameter in model.parameters()
        ]

        train_one_epoch(
            model=model,
            data_loader=loader,
            optimizer=optimizer,
            loss_function=nn.CrossEntropyLoss(),
        )

        parameters_after = list(model.parameters())

        self.assertTrue(
            any(
                not torch.equal(before, after.detach())
                for before, after in zip(parameters_before, parameters_after)
            )
        )


class TestCreateModel(unittest.TestCase):
    def test_creates_language_model(self):
        model = create_model(vocab_size=32)
        self.assertIsInstance(model, CoastMindLanguageModel)



class TestTrain(unittest.TestCase):
    def test_returns_language_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "coastmind.pt"
            model = train(checkpoint_path=checkpoint_path)

            self.assertIsInstance(model, CoastMindLanguageModel)
            self.assertTrue(checkpoint_path.exists())

    def test_saved_checkpoint_can_be_loaded(self):
        from src.training.checkpoint import load_checkpoint

        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "coastmind.pt"
            train(checkpoint_path=checkpoint_path)

            loaded_model, loaded_tokenizer = load_checkpoint(
                checkpoint_path
            )

            self.assertIsInstance(loaded_model, CoastMindLanguageModel)
            self.assertGreater(len(loaded_tokenizer.vocab), 0)



if __name__ == "__main__":
    unittest.main()
