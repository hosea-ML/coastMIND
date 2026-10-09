
# CoastMind Training-Data Pipeline

## 1. Purpose

The training-data pipeline connects corpus preparation with next-token sequence generation.

Implementation: `src/training/prepare_training_data.py`

Automated tests: `tests/test_prepare_training_data.py`

## 2. Pipeline Workflow

1. Load and validate the pilot corpus.
2. Combine the text records.
3. Convert the text into character-level token IDs.
4. Generate overlapping input and target sequences.
5. Return the prepared data for inspection and future model training.

## 3. Function

The `prepare_training_data()` function accepts a `block_size`, which defaults to 32.

It calls `prepare_corpus()` and passes the resulting token IDs to `create_input_target_pairs()`.

The function returns the corpus records, tokenizer, input tensor, target tensor and block size.

## 4. Verified Results

Using the current synthetic pilot corpus:

- Records: 3
- Characters: 492
- Vocabulary size: 32
- Block size: 32
- Training examples: 460
- Input tensor shape: (460, 32)
- Target tensor shape: (460, 32)
- Automated tests passed: 25

## 5. Input and Target Alignment

The target sequence is shifted one token ahead of the input sequence.

The model will eventually learn to predict the next token at each position.

## 6. Automated Tests

Run the complete test suite:

`python -m unittest discover -s tests -v`

The latest verified result is 25 passing tests.

## 7. Limitations

The current pilot contains only three synthetic records. The 460 overlapping windows are not independent documents and are not sufficient for meaningful scientific language-model training.

The pipeline does not yet implement minibatch sampling, dataset splitting, a Transformer, an optimizer or a training loop.

## 8. Next Milestone

Build and inspect the model's token embeddings and positional embeddings, which provide the numerical representations used by the Transformer.
