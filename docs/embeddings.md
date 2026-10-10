
# CoastMind Embedding Architecture

## 1. Purpose

The embedding module converts token IDs into numerical vector representations that can be processed by a Transformer.

Implementation: `src/model/embeddings.py`

Automated tests: `tests/test_embeddings.py`

## 2. Token Embedding

The `TokenEmbedding` class uses `torch.nn.Embedding` to map each token ID to a trainable vector.

For the current pilot vocabulary:

- Vocabulary size: 32
- Embedding dimension: 16
- Embedding table shape: (32, 16)
- Number of trainable parameters: 512

## 3. Positional Embedding

The `PositionalEmbedding` class assigns a trainable vector to each supported sequence position.

The current configuration supports 32 positions, with 16 values per position.

Embedding table shape: (32, 16)

The implementation uses learned positional embeddings.

## 4. Combined Embeddings

The `CoastMindEmbedding` class combines token and positional embeddings through element-wise addition.

For a batch of two sequences, each containing four tokens, with an embedding dimension of 16:

- Input shape: (2, 4)
- Token embedding shape: (2, 4, 16)
- Positional embedding shape: (1, 4, 16)
- Combined output shape: (2, 4, 16)

The positional embeddings are broadcast across the batch.

The implementation rejects sequences longer than the configured maximum length.

## 5. Trainable Parameters

Both embedding tables contain trainable parameters. During model training, their values can be updated through gradient-based optimization.

## 6. Automated Tests

Run the embedding tests:

`python -m unittest tests.test_embeddings -v`

Run the complete test suite:

`python -m unittest discover -s tests -v`

Latest verified result: 40 tests passed.

## 7. Limitations

The current implementation uses character-level tokens and learned positional embeddings.

It does not yet include attention, Transformer blocks, a language-model output layer, or a training loop.

The current synthetic pilot corpus is suitable for pipeline testing, not meaningful scientific language-model training.

## 8. Next Milestone

Implement and test causal self-attention, allowing each token to use information from earlier positions without accessing future tokens.
