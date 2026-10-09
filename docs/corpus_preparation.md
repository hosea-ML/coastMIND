# CoastMind Corpus Preparation Pipeline

## 1. Purpose

The corpus preparation module connects the dataset loader and character tokenizer to convert text records into numerical token IDs.

Implementation: `src/data/prepare_corpus.py`

Dependencies within CoastMind:
- `src/data/dataset.py`
- `src/data/tokenizer.py`

Input: `data/raw/pilot_corpus.jsonl`

## 2. Processing Steps

1. Load and validate the JSONL records.
2. Reject an empty corpus.
3. Combine record text using newline separators.
4. Build a character vocabulary from the combined text.
5. Encode the text into integer token IDs.
6. Decode the token IDs and compare the result with the original text.

## 3. Verified Pilot Results

| Measurement | Result |
|---|---:|
| Records loaded | 3 |
| Characters in combined text | 492 |
| Vocabulary size | 32 |
| Token IDs generated | 492 |
| Round-trip reconstruction | Successful |

These values describe the current pilot corpus and may change when its text changes.

## 4. Automated Testing

Run the complete test suite:

`python -m unittest discover -s tests -v`

Latest verified result: 16 tests passed.

The tests cover dataset loading, tokenizer behaviour, corpus preparation, vocabulary integrity, text reconstruction, and empty-corpus handling.

## 5. Limitations

The current pipeline uses character-level tokenization. Each character, including spaces, punctuation, and newline separators, becomes a token.

The pilot corpus contains only three synthetic educational records. It is suitable for testing software components, not for training a useful language model.

The pipeline does not yet implement train/validation/test splits, persistent tokenizer vocabulary files, special tokens, or model-ready sequence batching.

## 6. Next Milestone

The next planned stage is to create model-ready numerical sequences and understand how a language model learns to predict the next token.

Before meaningful training, the corpus must be expanded, its sources and licences reviewed, and an appropriate data-splitting strategy implemented.
