
"""Save and load CoastMind model checkpoints."""

from pathlib import Path

import torch

from src.data.tokenizer import CharacterTokenizer
from src.model.language_model import CoastMindLanguageModel


CHECKPOINT_VERSION = 1


def save_checkpoint(
    model: CoastMindLanguageModel,
    tokenizer: CharacterTokenizer,
    path: str | Path,
) -> None:
    """Save model weights, architecture configuration, and vocabulary."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    first_block = model.transformer_blocks[0]

    model_config = {
        "vocab_size": model.vocab_size,
        "embedding_dim": model.embedding.token_embedding.embedding.embedding_dim,
        "num_heads": first_block.attention.num_heads,
        "hidden_dim": first_block.feed_forward.network[0].out_features,
        "num_layers": len(model.transformer_blocks),
        "max_seq_length": model.max_seq_length,
    }

    checkpoint = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "model_config": model_config,
        "model_state_dict": model.state_dict(),
        "vocabulary": tokenizer.vocab,
    }

    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(
    path: str | Path,
) -> tuple[CoastMindLanguageModel, CharacterTokenizer]:
    """Load a checkpoint and reconstruct its model and tokenizer."""
    checkpoint_path = Path(path)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Unsupported checkpoint version.")

    model_config = checkpoint["model_config"]
    vocabulary = checkpoint["vocabulary"]

    if not vocabulary or len(set(vocabulary)) != len(vocabulary):
        raise ValueError("Checkpoint vocabulary is invalid.")

    if len(vocabulary) != model_config["vocab_size"]:
        raise ValueError("Vocabulary size does not match model configuration.")

    model = CoastMindLanguageModel(**model_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = CharacterTokenizer("".join(vocabulary))

    if tokenizer.vocab != vocabulary:
        raise ValueError("Checkpoint vocabulary order is invalid.")

    return model, tokenizer
