# CoastMind Character Tokenizer

## 1. Purpose

The character tokenizer converts text into numerical token IDs and reconstructs text from those IDs.

Implementation: `src/data/tokenizer.py`

Automated tests: `tests/test_tokenizer.py`

## 2. How It Works

The `CharacterTokenizer` class builds a vocabulary from the unique characters in its input text.

It creates two mappings:

- `stoi`: maps each character to an integer ID.
- `itos`: maps each integer ID back to its character.

The `encode()` method converts text into a sequence of integer IDs.

The `decode()` method reconstructs text from a sequence of IDs.

## 3. Example

For the input text `tide`, the sorted vocabulary is:

`['d', 'e', 'i', 't']`

The character-to-ID mapping is:

- `d` → 0
- `e` → 1
- `i` → 2
- `t` → 3

Therefore:

- Text: `tide`
- Token IDs: `[3, 2, 0, 1]`
- Decoded text: `tide`

## 4. Automated Tests

Run the complete test suite with:

`python -m unittest discover -s tests -v`

Latest verified result: 12 tests passed, comprising five dataset-loader tests and seven tokenizer tests.

The tokenizer tests cover vocabulary construction, encoding, decoding, round-trip reconstruction, unknown characters, empty training text, and invalid token IDs.

## 5. Current Limitations

This is a character-level baseline tokenizer.

- It treats each character as a separate token.
- Longer passages therefore produce longer token sequences.
- Its vocabulary is limited to characters in the text used to construct it.
- It rejects unknown characters and invalid token IDs.

The vocabulary must eventually be built from a suitably prepared corpus rather than a single example.

## 6. Next Steps

1. Test the tokenizer with CoastMind's pilot corpus.
2. Decide how to handle unseen characters and special tokens.
3. Build a tokenizer from the prepared corpus.
4. Convert corpus text into token IDs.
5. Compare character-level tokenization with a subword approach.
