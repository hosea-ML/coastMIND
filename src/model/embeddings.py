
import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Convert token IDs into trainable vector representations."""

    def __init__(self, vocab_size: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(token_ids)


class PositionalEmbedding(nn.Module):
    """Provide trainable representations of token positions."""

    def __init__(self, max_seq_length: int, embedding_dim: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=max_seq_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(position_ids)

class CoastMindEmbedding(nn.Module):
    """Combine token and positional embeddings."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        max_seq_length: int,
    ):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
        )

        self.position_embedding = PositionalEmbedding(
            max_seq_length=max_seq_length,
            embedding_dim=embedding_dim,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        sequence_length = token_ids.shape[1]

        if sequence_length > self.position_embedding.embedding.num_embeddings:
            raise ValueError("Sequence length exceeds max_seq_length.")

        position_ids = torch.arange(
            sequence_length,
            device=token_ids.device,
        ).unsqueeze(0)

        token_vectors = self.token_embedding(token_ids)
        position_vectors = self.position_embedding(position_ids)

        return token_vectors + position_vectors
