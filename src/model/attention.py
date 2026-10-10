
import torch
import math
import torch.nn as nn


class QKVProjection(nn.Module):
    """Project input embeddings into queries, keys, and values."""

    def __init__(self, embedding_dim: int):
        super().__init__()

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim, embedding_dim)
        self.value = nn.Linear(embedding_dim, embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)

        return queries, keys, values

class AttentionScores(nn.Module):
    """Calculate attention scores from queries and keys."""

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
    ) -> torch.Tensor:
        keys_transposed = keys.transpose(-2, -1)

        scores = torch.matmul(queries, keys_transposed)

        return scores

class AttentionScaling(nn.Module):
    """Scale attention scores by the square root of the key dimension."""

    def __init__(self, key_dim: int):
        super().__init__()

        if key_dim <= 0:
            raise ValueError("key_dim must be greater than zero.")

        self.scale = math.sqrt(key_dim)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return scores / self.scale

class AttentionSoftmax(nn.Module):
    """Convert attention scores into normalized weights."""

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        return torch.softmax(scores, dim=-1)

class CausalMask(nn.Module):
    """Create a mask that blocks attention to future token positions."""

    def forward(
        self,
        sequence_length: int,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be greater than zero.")

        mask = torch.tril(
            torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=device,
            )
        )

        return mask

class ValueAggregation(nn.Module):
    """Combine value vectors using attention weights."""

    def forward(
        self,
        attention_weights: torch.Tensor,
        values: torch.Tensor,
    ) -> torch.Tensor:
        context = torch.matmul(attention_weights, values)
        return context

class SelfAttention(nn.Module):
    """A single-head causal self-attention mechanism."""

    def __init__(self, embedding_dim: int):
        super().__init__()

        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be greater than zero.")

        self.qkv_projection = QKVProjection(embedding_dim)
        self.attention_scores = AttentionScores()
        self.attention_scaling = AttentionScaling(embedding_dim)
        self.causal_mask = CausalMask()
        self.attention_softmax = AttentionSoftmax()
        self.value_aggregation = ValueAggregation()

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        queries, keys, values = self.qkv_projection(x)

        scores = self.attention_scores(queries, keys)
        scaled_scores = self.attention_scaling(scores)

        mask = self.causal_mask(
            sequence_length=x.size(1),
            device=x.device,
        )

        masked_scores = scaled_scores.masked_fill(
            ~mask,
            float("-inf"),
        )

        attention_weights = self.attention_softmax(masked_scores)

        context = self.value_aggregation(
            attention_weights,
            values,
        )

        if return_attention:
            return context, attention_weights

        return context
