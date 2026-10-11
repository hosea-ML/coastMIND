# CoastMind Language Model

## 1. Purpose

The CoastMind language model combines token embeddings, positional embeddings, Transformer blocks, and a language-model output head to predict the next token in a sequence.

It is a small, decoder-only Transformer implemented in PyTorch for educational experimentation with coastal-science text.

## 2. Architecture

The model consists of four main stages:

1. **Embedding layer:** `CoastMindEmbedding` converts token IDs into learned vectors and adds positional information.
2. **Transformer blocks:** A configurable number of `TransformerBlock` instances process the sequence using causal multi-head self-attention and feed-forward networks.
3. **Language-model head:** A linear layer maps the final hidden representation to one score per vocabulary token.
4. **Logits output:** The model returns raw scores for every token position. These scores can be converted into probabilities using softmax when needed.

## 3. Model implementation

The main class is `CoastMindLanguageModel`, implemented in `src/model/language_model.py`.

Its constructor accepts:

- `vocab_size`: Number of distinct tokens.
- `embedding_dim`: Size of each token representation.
- `num_heads`: Number of attention heads in each Transformer block.
- `hidden_dim`: Internal dimension of each feed-forward network.
- `num_layers`: Number of Transformer blocks.
- `max_seq_length`: Maximum supported sequence length.

## 4. Input and output shapes

The model expects token IDs with shape:

`(batch_size, sequence_length)`

It returns logits with shape:

`(batch_size, sequence_length, vocab_size)`

For example, an input with shape `(2, 4)` and a vocabulary size of 32 produces logits with shape `(2, 4, 32)`.

Each position produces a score for every possible next token. During training, these scores must be aligned with the corresponding next-token targets.

## 5. Causal attention

The attention layers use a causal mask to prevent a token position from attending to future positions. This supports autoregressive next-token prediction.

The model does not need a separate causal mask in its output head because causal masking is applied inside the attention mechanism.

## 6. Testing

The language-model test suite checks output shape, tensor type, finite logits, the number of Transformer blocks, gradient flow, and rejection of zero vocabulary size and zero Transformer layers.

The full project test suite currently contains 101 tests, all of which passed in the recorded test run.

## 7. Current limitations

- The current character-level tokenizer uses a small vocabulary.
- The pilot corpus contains only three synthetic records and is intended for pipeline verification, not meaningful scientific training.
- A successful forward pass and passing tests do not establish predictive accuracy.
- Training, evaluation, and autoregressive text generation remain separate stages of development.
- The model's input validation can be strengthened before the training pipeline is built.

## 8. Next development milestone

Implement a reproducible training loop using shifted next-token targets, cross-entropy loss, an optimizer such as AdamW, and a small CPU-friendly configuration. Evaluate the training pipeline before attempting to interpret generated coastal-science text.