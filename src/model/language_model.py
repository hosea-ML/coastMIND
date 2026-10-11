
import torch
import torch.nn as nn

from src.model.embeddings import CoastMindEmbedding
from src.model.transformer_block import TransformerBlock


class CoastMindLanguageModel(nn.Module):
    """A small decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        max_seq_length: int,
    ):
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size must be greater than zero.")

        if num_layers <= 0:
            raise ValueError("num_layers must be greater than zero.")

        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        self.embedding = CoastMindEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            max_seq_length=max_seq_length,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                )
                for _ in range(num_layers)
            ]
        )

        self.language_model_head = nn.Linear(
            embedding_dim,
            vocab_size,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(token_ids)

        for block in self.transformer_blocks:
            x = block(x)

        logits = self.language_model_head(x)

        return logits
