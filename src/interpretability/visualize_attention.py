
"""Visualise actual CoastMind attention weights."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.interpretability.attention import get_attention_weights
from src.training.checkpoint import load_checkpoint


def plot_attention_heatmap(
    prompt: str = "coastal",
    block_index: int = 0,
    head_index: int = 0,
    output_path: str = "outputs/attention_heatmap.png",
) -> Path:
    """Save a heatmap of a selected attention head."""
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    model, tokenizer = load_checkpoint("outputs/coastmind_checkpoint.pt")

    token_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
    )

    attention_weights = get_attention_weights(
        model,
        token_ids,
        block_index=block_index,
    )

    if not 0 <= head_index < attention_weights.shape[1]:
        raise ValueError(
            f"head_index must be between 0 and "
            f"{attention_weights.shape[1] - 1}."
        )

    matrix = attention_weights[0, head_index].cpu().numpy()
    characters = list(prompt)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 7))

    sns.heatmap(
        matrix,
        xticklabels=characters,
        yticklabels=characters,
        cmap="viridis",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        square=True,
    )

    plt.title(
        f"CoastMind Attention Heatmap\n"
        f"Block {block_index}, Head {head_index}"
    )
    plt.xlabel("Key position (character being attended to)")
    plt.ylabel("Query position (character doing the attending)")
    plt.tight_layout()
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()

    return output_file

def plot_all_attention_heads(
    prompt: str = "coastal",
    block_index: int = 0,
    output_path: str = "outputs/attention_heads_comparison.png",
) -> Path:
    """Save a side-by-side comparison of all attention heads."""
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    model, tokenizer = load_checkpoint("outputs/coastmind_checkpoint.pt")

    token_ids = torch.tensor(
        [tokenizer.encode(prompt)],
        dtype=torch.long,
    )

    attention_weights = get_attention_weights(
        model,
        token_ids,
        block_index=block_index,
    )

    number_of_heads = attention_weights.shape[1]
    matrices = attention_weights[0].cpu().numpy()
    characters = list(prompt)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(
        1,
        number_of_heads,
        figsize=(5 * number_of_heads, 5),
        constrained_layout=True,
    )

    for head_index, axis in enumerate(axes):
        sns.heatmap(
            matrices[head_index],
            xticklabels=characters,
            yticklabels=characters,
            cmap="viridis",
            vmin=0,
            vmax=1,
            annot=True,
            fmt=".2f",
            square=True,
            ax=axis,
            cbar=(head_index == number_of_heads - 1),
        )

        axis.set_title(f"Head {head_index}")
        axis.set_xlabel("Key position")
        axis.set_ylabel("Query position")

    figure.suptitle(
        f"CoastMind Attention Heads - Block {block_index}\n"
        f"Prompt: {prompt}"
    )
    figure.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(figure)

    return output_file

if __name__ == "__main__":
    saved_path = plot_all_attention_heads()
    print(f"Attention heads comparison saved to: {saved_path.resolve()}")