
from pathlib import Path

from src.data.dataset import load_corpus
from src.data.tokenizer import CharacterTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "raw" / "pilot_corpus.jsonl"


def prepare_corpus(
    path: str | Path = DEFAULT_CORPUS_PATH,
) -> dict:
    """Load text records and encode their combined text."""

    records = load_corpus(path)

    if not records:
        raise ValueError("Cannot prepare an empty corpus.")

    # Separate records with newlines to preserve their boundaries.
    corpus_text = "\n".join(record["text"] for record in records)

    # Build one vocabulary from all the combined corpus text.
    tokenizer = CharacterTokenizer(corpus_text)

    # Convert every character into its corresponding integer ID.
    token_ids = tokenizer.encode(corpus_text)

    return {
        "records": records,
        "text": corpus_text,
        "tokenizer": tokenizer,
        "token_ids": token_ids,
    }


def main() -> None:
    """Prepare the pilot corpus and print a summary."""

    result = prepare_corpus()

    print("Corpus preparation successful")
    print("Number of records:", len(result["records"]))
    print("Characters in corpus:", len(result["text"]))
    print("Vocabulary size:", len(result["tokenizer"].vocab))
    print("Number of token IDs:", len(result["token_ids"]))
    print("First 30 token IDs:", result["token_ids"][:30])

    reconstructed_text = result["tokenizer"].decode(result["token_ids"])
    print("Round-trip successful:", reconstructed_text == result["text"])


if __name__ == "__main__":
    main()
