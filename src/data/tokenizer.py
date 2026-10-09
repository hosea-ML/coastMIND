
class CharacterTokenizer:
    """A simple character-level tokenizer."""

    def __init__(self, text: str):
        if not text:
            raise ValueError("Training text cannot be empty.")

        self.vocab = sorted(set(text))
        self.stoi = {
            character: index
            for index, character in enumerate(self.vocab)
        }
        self.itos = {
            index: character
            for character, index in self.stoi.items()
        }

    def encode(self, text: str) -> list[int]:
        """Convert text into a list of token IDs."""
        unknown = set(text) - set(self.stoi)

        if unknown:
            raise ValueError(
                f"Text contains unknown characters: {sorted(unknown)}"
            )

        return [self.stoi[character] for character in text]

    def decode(self, token_ids: list[int]) -> str:
        """Convert token IDs back into text."""
        try:
            return "".join(self.itos[token_id] for token_id in token_ids)
        except KeyError as exc:
            raise ValueError(
                f"Unknown token ID: {exc.args[0]}"
            ) from exc
