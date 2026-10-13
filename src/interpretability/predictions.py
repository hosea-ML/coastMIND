
"""Tools for inspecting next-token predictions in CoastMind."""

import torch

from src.data.tokenizer import CharacterTokenizer
from src.model.language_model import CoastMindLanguageModel


def analyze_next_token(
    model: CoastMindLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    top_k: int = 5,
    temperature: float = 1.0,
) -> list[dict[str, float | int | str]]:
    """Rank next-character predictions for a prompt.

    Returns the top-k candidates, their raw logits, and their
    probabilities after temperature scaling.
    """
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be greater than zero.")

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    if not torch.isfinite(torch.tensor(temperature)):
        raise ValueError("temperature must be finite.")

    token_ids = tokenizer.encode(prompt)

    if len(token_ids) > model.max_seq_length:
        raise ValueError(
            "The input sequence exceeds the model's maximum sequence length."
        )

    input_ids = torch.tensor([token_ids], dtype=torch.long)

    model.eval()

    with torch.no_grad():
        logits = model(input_ids)
        next_token_logits = logits[0, -1, :]
        probabilities = torch.softmax(
            next_token_logits / temperature,
            dim=-1,
        )

        candidate_count = min(top_k, model.vocab_size)
        top_probabilities, top_token_ids = torch.topk(
            probabilities,
            k=candidate_count,
        )

    results = []

    for token_id, probability in zip(
        top_token_ids.tolist(),
        top_probabilities.tolist(),
    ):
        results.append(
            {
                "token_id": token_id,
                "token": tokenizer.itos[token_id],
                "logit": next_token_logits[token_id].item(),
                "probability": probability,
            }
        )

    return results
