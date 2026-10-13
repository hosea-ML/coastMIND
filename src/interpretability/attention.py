
"""Tools for inspecting attention weights in CoastMind."""

import torch

from src.model.language_model import CoastMindLanguageModel


def get_attention_weights(
    model: CoastMindLanguageModel,
    token_ids: torch.Tensor,
    block_index: int = 0,
) -> torch.Tensor:
    """Return attention weights from a selected block's real forward pass.

    Args:
        model: A loaded CoastMind language model.
        token_ids: Token IDs shaped (batch_size, sequence_length).
        block_index: Zero-based Transformer block index.

    Returns:
        Attention weights shaped
        (batch_size, num_heads, sequence_length, sequence_length).
    """
    if token_ids.ndim != 2:
        raise ValueError(
            "token_ids must have shape (batch_size, sequence_length)."
        )

    if token_ids.shape[1] == 0:
        raise ValueError("token_ids cannot contain an empty sequence.")

    if not 0 <= block_index < len(model.transformer_blocks):
        raise ValueError(
            f"block_index must be between 0 and "
            f"{len(model.transformer_blocks) - 1}."
        )

    if token_ids.shape[1] > model.max_seq_length:
        raise ValueError(
            "The input sequence exceeds the model's maximum sequence length."
        )

    model.eval()

    with torch.no_grad():
        x = model.embedding(token_ids)

        for index, block in enumerate(model.transformer_blocks):
            if index == block_index:
                _, attention_weights = block.attention(
                    x,
                    return_attention=True,
                )

                return attention_weights

            x = block(x)

    raise RuntimeError("Unable to extract attention weights.")
