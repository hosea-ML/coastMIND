
"""Visualize CoastMind next-token predictions with Plotly."""

from pathlib import Path

import plotly.express as px

from src.interpretability.predictions import analyze_next_token
from src.training.checkpoint import load_checkpoint


def format_token_label(token: str) -> str:
    """Make whitespace tokens easy to identify in charts."""
    special_labels = {
        " ": "SPACE",
        "\t": "TAB",
        "\n": "NEWLINE",
        "\r": "CARRIAGE RETURN",
    }

    return special_labels.get(token, repr(token))


def plot_next_token_predictions(
    prompt: str = "coastal",
    top_k: int = 5,
    temperature: float = 1.0,
    checkpoint_path: str | Path = "outputs/coastmind_checkpoint.pt",
    output_path: str | Path = "outputs/next_token_predictions.html",
) -> Path:
    """Create and save an interactive next-token probability chart."""

    model, tokenizer = load_checkpoint(checkpoint_path)

    predictions = analyze_next_token(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        top_k=top_k,
        temperature=temperature,
    )

    labels = [
        format_token_label(item["token"])
        for item in predictions
    ]

    probabilities = [
        item["probability"]
        for item in predictions
    ]

    logits = [
        item["logit"]
        for item in predictions
    ]

    figure = px.bar(
        x=labels,
        y=probabilities,
        hover_data={"Raw logit": logits},
        labels={
            "x": "Candidate next character",
            "y": "Probability",
        },
        title=(
            "CoastMind Next-Character Predictions"
            f"<br>Prompt: {prompt!r} | "
            f"Temperature: {temperature} | Top-k: {len(predictions)}"
        ),
    )

    figure.update_layout(
        yaxis_tickformat=".1%",
        yaxis_range=[0, 1],
        template="plotly_white",
    )

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    figure.write_html(destination, include_plotlyjs=True)

    return destination


if __name__ == "__main__":
    saved_path = plot_next_token_predictions()
    print(f"Prediction chart saved to: {saved_path.resolve()}")
