
import unittest

from src.data.tokenizer import CharacterTokenizer


class TestCharacterTokenizer(unittest.TestCase):
    def test_builds_sorted_vocabulary(self):
        tokenizer = CharacterTokenizer("tide")

        self.assertEqual(tokenizer.vocab, ["d", "e", "i", "t"])

    def test_encodes_text(self):
        tokenizer = CharacterTokenizer("tide")

        self.assertEqual(tokenizer.encode("tide"), [3, 2, 0, 1])

    def test_decodes_token_ids(self):
        tokenizer = CharacterTokenizer("tide")

        self.assertEqual(tokenizer.decode([3, 2, 0, 1]), "tide")

    def test_encode_decode_round_trip(self):
        text = "coastal science"
        tokenizer = CharacterTokenizer(text)

        self.assertEqual(tokenizer.decode(tokenizer.encode(text)), text)

    def test_rejects_unknown_characters(self):
        tokenizer = CharacterTokenizer("tide")

        with self.assertRaisesRegex(ValueError, "unknown characters"):
            tokenizer.encode("tides")

    def test_rejects_empty_training_text(self):
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            CharacterTokenizer("")

    def test_rejects_unknown_token_ids(self):
        tokenizer = CharacterTokenizer("tide")

        with self.assertRaisesRegex(ValueError, "Unknown token ID"):
            tokenizer.decode([99])


if __name__ == "__main__":
    unittest.main()
