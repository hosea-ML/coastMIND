# CoastMind Next-Token Sequence Generation

## 1. Purpose

The sequence-generation module converts a sequence of token IDs into aligned input and target tensors for next-token prediction.

Implementation: `src/training/sequence.py`

Automated tests: `tests/test_sequence.py`

## 2. How It Works

The `create_input_target_pairs()` function accepts a list of token IDs and a block size.

It creates overlapping input sequences and corresponding target sequences. Each target is shifted one position to the right relative to its input.

For example, given:

`[4, 7, 2, 9, 5]`

With a block size of four:

- Input: `[4, 7, 2, 9]`
- Target: `[7, 2, 9, 5]`

Each input token position is associated with the next token as its prediction target.

## 3. PyTorch Tensors

The function returns two tensors:

- `x`: input token IDs.
- `y`: target token IDs.

Both use the `torch.long` data type.

Their shape is `(number_of_examples, block_size)`.

## 4. Sliding Windows

For a longer token sequence, the function moves the input window forward one token at a time.

This creates overlapping training examples and allows the model to learn from multiple positions in the sequence.

The current implementation creates all possible windows of the requested size from the supplied sequence.

## 5. Automated Tests

Run the complete test suite:

`python -m unittest discover -s tests -v`

Latest verified result: 21 tests passed.

Five tests cover sequence generation, sliding windows, tensor types, invalid block sizes, and insufficient input length.

## 6. Limitations

The function currently creates every possible overlapping window in memory. This is acceptable for small examples but may be inefficient for a large corpus.

The module does not yet implement minibatch sampling, train/validation/test splitting, or model training.

## 7. Next Milestone

The next stage is to connect the prepared corpus to the sequence generator and learn how a decoder-only Transformer uses these input-target pairs to predict the next token.
