
"""Training engine for the CoastMind language model."""

import torch
from pathlib import Path
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset

from configs import training_config as config
from src.model.language_model import CoastMindLanguageModel
from src.training.prepare_training_data import prepare_training_data
from src.training.checkpoint import save_checkpoint


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    torch.manual_seed(seed)


def create_model(vocab_size: int) -> CoastMindLanguageModel:
    """Create a CoastMind model from the training configuration."""
    return CoastMindLanguageModel(
        vocab_size=vocab_size,
        embedding_dim=config.EMBEDDING_DIM,
        num_heads=config.NUM_HEADS,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=config.NUM_LAYERS,
        max_seq_length=config.BLOCK_SIZE,
    )


def create_data_loader(
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> DataLoader:
    """Group input-target pairs into mini-batches."""
    dataset = TensorDataset(inputs, targets)

    return DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
    )


def train_one_epoch(
    model: CoastMindLanguageModel,
    data_loader: DataLoader,
    optimizer: AdamW,
    loss_function: nn.Module,
) -> float:
    """Train the model for one epoch and return average loss."""
    model.train()

    total_loss = 0.0
    total_examples = 0

    for inputs, targets in data_loader:
        optimizer.zero_grad()

        logits = model(inputs)

        loss = loss_function(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size

    return total_loss / total_examples


def train(
    checkpoint_path: str | Path = "outputs/coastmind_checkpoint.pt",
) -> CoastMindLanguageModel:
    """Prepare data, build the model, and train it."""
    set_random_seed(config.RANDOM_SEED)

    training_data = prepare_training_data(
        block_size=config.BLOCK_SIZE,
    )

    tokenizer = training_data["tokenizer"]
    vocab_size = len(tokenizer.itos)

    model = create_model(vocab_size)

    data_loader = create_data_loader(
        inputs=training_data["inputs"],
        targets=training_data["targets"],
    )

    optimizer = AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
    )

    loss_function = nn.CrossEntropyLoss()

    print("Starting CoastMind training")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Training examples: {len(training_data['inputs'])}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    print()

    for epoch in range(config.NUM_EPOCHS):
        average_loss = train_one_epoch(
            model=model,
            data_loader=data_loader,
            optimizer=optimizer,
            loss_function=loss_function,
        )

        print(
            f"Epoch {epoch + 1}/{config.NUM_EPOCHS} "
            f"| Loss: {average_loss:.4f}"
        )

   
    save_checkpoint(
        model=model,
        tokenizer=tokenizer,
        path=checkpoint_path,
    )

    print(f"Checkpoint saved to: {checkpoint_path}")
    print("\nCoastMind training completed.")

    return model


if __name__ == "__main__":
    train()
