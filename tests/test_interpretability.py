
"""Tests for CoastMind interpretability tools."""

import pytest
import torch

from src.interpretability.attention import get_attention_weights
from src.model.language_model import CoastMindLanguageModel


@pytest.fixture
def model_and_token_ids():
    """Create a small model and a sample token sequence."""
    torch.manual_seed(42)

    model = CoastMindLanguageModel(
        vocab_size=8,
        embedding_dim=16,
        num_heads=4,
        hidden_dim=32,
        num_layers=2,
        max_seq_length=16,
    )

    token_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

    return model, token_ids


@pytest.mark.parametrize("block_index", [0, 1])
def test_attention_weights_have_expected_shape(
    model_and_token_ids,
    block_index,
):
    model, token_ids = model_and_token_ids

    weights = get_attention_weights(model, token_ids, block_index)

    assert weights.shape == (1, 4, 4, 4)


@pytest.mark.parametrize("block_index", [0, 1])
def test_attention_weights_are_valid_probabilities(
    model_and_token_ids,
    block_index,
):
    model, token_ids = model_and_token_ids

    weights = get_attention_weights(model, token_ids, block_index)

    row_sums = weights.sum(dim=-1)

    assert torch.allclose(
        row_sums,
        torch.ones_like(row_sums),
        atol=1e-5,
    )


@pytest.mark.parametrize("block_index", [0, 1])
def test_attention_cannot_see_future_positions(
    model_and_token_ids,
    block_index,
):
    model, token_ids = model_and_token_ids

    weights = get_attention_weights(model, token_ids, block_index)

    future_weights = torch.triu(weights, diagonal=1)

    assert torch.count_nonzero(future_weights).item() == 0


def test_invalid_block_index_raises_error(model_and_token_ids):
    model, token_ids = model_and_token_ids

    with pytest.raises(ValueError, match="block_index"):
        get_attention_weights(model, token_ids, block_index=2)


def test_one_dimensional_token_ids_raise_error(model_and_token_ids):
    model, token_ids = model_and_token_ids

    with pytest.raises(ValueError, match="batch_size"):
        get_attention_weights(model, token_ids[0])


def test_empty_sequence_raises_error(model_and_token_ids):
    model, _ = model_and_token_ids

    empty_token_ids = torch.empty((1, 0), dtype=torch.long)

    with pytest.raises(ValueError, match="empty sequence"):
        get_attention_weights(model, empty_token_ids)


def test_sequence_exceeding_model_limit_raises_error(model_and_token_ids):
    model, _ = model_and_token_ids

    token_ids = torch.ones((1, 17), dtype=torch.long)

    with pytest.raises(ValueError, match="maximum sequence length"):
        get_attention_weights(model, token_ids)
