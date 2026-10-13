
"""Tests for CoastMind prediction visualization."""

from pathlib import Path
from unittest.mock import patch

from src.interpretability.visualize_predictions import (
    format_token_label,
    plot_next_token_predictions,
)


def test_whitespace_tokens_have_readable_labels():
    assert format_token_label(" ") == "SPACE"
    assert format_token_label("\t") == "TAB"
    assert format_token_label("\n") == "NEWLINE"
    assert format_token_label("\r") == "CARRIAGE RETURN"


def test_regular_character_label_uses_repr():
    assert format_token_label("a") == "'a'"


@patch("src.interpretability.visualize_predictions.load_checkpoint")
@patch("src.interpretability.visualize_predictions.analyze_next_token")
def test_creates_html_chart(
    mock_analyze,
    mock_load_checkpoint,
    tmp_path,
):
    mock_load_checkpoint.return_value = (object(), object())
    mock_analyze.return_value = [
        {
            "token_id": 0,
            "token": " ",
            "logit": 1.5,
            "probability": 0.7,
        },
        {
            "token_id": 1,
            "token": "a",
            "logit": 0.5,
            "probability": 0.3,
        },
    ]

    output_path = tmp_path / "chart.html"

    result = plot_next_token_predictions(
        prompt="coastal",
        top_k=2,
        temperature=0.8,
        checkpoint_path=tmp_path / "checkpoint.pt",
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.is_file()
    assert output_path.stat().st_size > 0

    html = output_path.read_text(encoding="utf-8")
    assert "CoastMind Next-Character Predictions" in html
    assert "coastal" in html
    assert "0.8" in html
    assert "SPACE" in html

    mock_load_checkpoint.assert_called_once_with(
        tmp_path / "checkpoint.pt"
    )


@patch("src.interpretability.visualize_predictions.load_checkpoint")
@patch("src.interpretability.visualize_predictions.analyze_next_token")
def test_forwards_prediction_parameters(
    mock_analyze,
    mock_load_checkpoint,
    tmp_path,
):
    mock_load_checkpoint.return_value = (object(), object())
    mock_analyze.return_value = [
        {
            "token_id": 0,
            "token": "a",
            "logit": 1.0,
            "probability": 1.0,
        }
    ]

    plot_next_token_predictions(
        prompt="shore",
        top_k=1,
        temperature=0.5,
        checkpoint_path=tmp_path / "checkpoint.pt",
        output_path=tmp_path / "chart.html",
    )

    mock_analyze.assert_called_once_with(
        model=mock_load_checkpoint.return_value[0],
        tokenizer=mock_load_checkpoint.return_value[1],
        prompt="shore",
        top_k=1,
        temperature=0.5,
    )
