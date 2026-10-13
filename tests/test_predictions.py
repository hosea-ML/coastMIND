
"""Tests for CoastMind next-token prediction analysis."""

import pytest
import torch

from src.data.tokenizer import CharacterTokenizer
from src.interpretability.predictions import analyze_next_token
from src.model.language_model import CoastMindLanguageModel


@pytest.fixture
def model_and_tokenizer():
    """Create a small model and matching character tokenizer."""
    torch.manual_seed(42)

    tokenizer = CharacterTokenizer("abcdefgh")

    model = CoastMindLanguageModel(
        vocab_size=len(tokenizer.vocab),
        embedding_dim=16,
        num_heads=4,
        hidden_dim=32,
        num_layers=2,
        max_seq_length=16,
    )

    return model, tokenizer


def test_returns_requested_number_of_candidates(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    results = analyze_next_token(
        model, tokenizer, prompt="abc", top_k=3
    )

    assert len(results) == 3


def test_candidates_are_sorted_by_probability(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    results = analyze_next_token(
        model, tokenizer, prompt="abc", top_k=5
    )

    probabilities = [item["probability"] for item in results]

    assert probabilities == sorted(probabilities, reverse=True)


def test_probabilities_sum_to_one_for_full_vocabulary(
    model_and_tokenizer,
):
    model, tokenizer = model_and_tokenizer

    results = analyze_next_token(
        model,
        tokenizer,
        prompt="abc",
        top_k=len(tokenizer.vocab),
    )

    total = sum(item["probability"] for item in results)

    assert total == pytest.approx(1.0, abs=1e-6)


def test_results_include_expected_fields(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    results = analyze_next_token(
        model, tokenizer, prompt="abc", top_k=1
    )

    assert set(results[0]) == {
        "token_id",
        "token",
        "logit",
        "probability",
    }


def test_top_k_larger_than_vocabulary_returns_all_tokens(
    model_and_tokenizer,
):
    model, tokenizer = model_and_tokenizer

    results = analyze_next_token(
        model,
        tokenizer,
        prompt="abc",
        top_k=len(tokenizer.vocab) + 5,
    )

    assert len(results) == len(tokenizer.vocab)


def test_empty_prompt_raises_error(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    with pytest.raises(ValueError, match="Prompt cannot be empty"):
        analyze_next_token(model, tokenizer, prompt="")


@pytest.mark.parametrize("top_k", [0, -1])
def test_invalid_top_k_raises_error(model_and_tokenizer, top_k):
    model, tokenizer = model_and_tokenizer

    with pytest.raises(ValueError, match="top_k"):
        analyze_next_token(
            model, tokenizer, prompt="abc", top_k=top_k
        )


@pytest.mark.parametrize("temperature", [0, -1])
def test_invalid_temperature_raises_error(
    model_and_tokenizer, temperature
):
    model, tokenizer = model_and_tokenizer

    with pytest.raises(ValueError, match="temperature"):
        analyze_next_token(
            model,
            tokenizer,
            prompt="abc",
            temperature=temperature,
        )


@pytest.mark.parametrize("temperature", [float("inf"), float("nan")])
def test_non_finite_temperature_raises_error(
    model_and_tokenizer, temperature
):
    model, tokenizer = model_and_tokenizer

    with pytest.raises(ValueError, match="temperature"):
        analyze_next_token(
            model,
            tokenizer,
            prompt="abc",
            temperature=temperature,
        )


def test_unknown_character_raises_error(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer

    with pytest.raises(ValueError, match="unknown characters"):
        analyze_next_token(model, tokenizer, prompt="abc?")


def test_sequence_exceeding_context_limit_raises_error(
    model_and_tokenizer,
):
    model, tokenizer = model_and_tokenizer

    with pytest.raises(ValueError, match="maximum sequence length"):
        analyze_next_token(
            model, tokenizer, prompt="a" * 17
        )
