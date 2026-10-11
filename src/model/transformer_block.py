
import torch
import torch.nn as nn
from src.model.attention import MultiHeadSelfAttention


class FeedForward(nn.Module):
    """Position-wise feed-forward neural network."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero.")

        self.network = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class TransformerBlock(nn.Module):
    """A decoder-style Transformer block with causal attention."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        )

        self.feed_forward = FeedForward(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
        )

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_output = self.attention(x)
        x = self.norm1(x + attention_output)

        feed_forward_output = self.feed_forward(x)
        x = self.norm2(x + feed_forward_output)

        return x
