
"""Generate text using a trained CoastMind language model."""

import torch

from src.data.tokenizer import CharacterTokenizer
from src.model.language_model import CoastMindLanguageModel


def generate_text(
    model: CoastMindLanguageModel,
    tokenizer: CharacterTokenizer,
    prompt: str,
    max_new_tokens: int = 50,
    temperature: float = 1.0,
) -> str:
    """Generate text autoregressively from a starting prompt."""

    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    if max_new_tokens < 0:
        raise ValueError("max_new_tokens cannot be negative.")

    if temperature <= 0:
        raise ValueError("temperature must be greater than zero.")

    model.eval()

    token_ids = tokenizer.encode(prompt)
    generated_ids = torch.tensor([token_ids], dtype=torch.long)

    for _ in range(max_new_tokens):
        # Keep the input within the model's context window.
        context = generated_ids[:, -model.max_seq_length:]

        with torch.no_grad():
            logits = model(context)

        # Select the predictions for the final input position.
        next_token_logits = logits[:, -1, :] / temperature

        # Convert logits into probabilities.
        probabilities = torch.softmax(next_token_logits, dim=-1)

        # Sample the next character.
        next_token = torch.multinomial(probabilities, num_samples=1)

        # Append the new token to the generated sequence.
        generated_ids = torch.cat(
            [generated_ids, next_token],
            dim=1,
        )

    return tokenizer.decode(generated_ids[0].tolist())
