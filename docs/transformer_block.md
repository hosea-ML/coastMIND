
# CoastMind Transformer Block

## 1. Overview

The Transformer block combines multi-head causal self-attention,
residual connections, layer normalization, and a position-wise
feed-forward neural network.

Implementation:
- `src/model/transformer_block.py`
- `src/model/attention.py`

Tests:
- `tests/test_transformer_block.py`

## 2. Feed-Forward Network

The `FeedForward` class applies three operations:

1. Linear projection from `embedding_dim` to `hidden_dim`.
2. GELU activation.
3. Linear projection back to `embedding_dim`.

For example, with `embedding_dim=8` and `hidden_dim=32`,
the feature dimensions are:

`8 → 32 → 8`

The network transforms features independently at each token position.

## 3. Transformer Block Architecture

The `TransformerBlock` class uses a post-layer-normalization design.

First, multi-head causal self-attention calculates contextual
representations. Its output is added to the original input and
normalized:

`x = LayerNorm(x + Attention(x))`

Next, the feed-forward network transforms the normalized
representations. A second residual connection and normalization
produce the final output:

`x = LayerNorm(x + FeedForward(x))`

The output retains the same shape as the input.

## 4. Tensor Dimensions

For an input of shape `(2, 4, 8)`:

- Batch size: 2
- Sequence length: 4
- Embedding dimension: 8
- Number of attention heads: 4
- Hidden dimension: 32
- Output shape: `(2, 4, 8)`

## 5. Testing

The test module contains 12 tests:

- Seven tests for `FeedForward`.
- Five tests for `TransformerBlock`.

These test output shapes, tensor types, finite values,
trainable parameters, gradient propagation, and invalid
feed-forward dimensions.

The full CoastMind test suite passed 94 tests after
implementing the Transformer block.

## 6. Limitations and Next Steps

Passing tests confirms the expected behaviour of the
implemented components for the tested cases. It does not
demonstrate language-model quality or scientific accuracy.

The next milestone is to build a decoder-only language model
that combines token embeddings, positional embeddings,
one or more Transformer blocks, and a language-model output head.
