
from src.data.prepare_corpus import prepare_corpus
from src.training.sequence import create_input_target_pairs


def prepare_training_data(block_size: int = 32) -> dict:
    """Prepare token sequences for next-token prediction."""

    corpus = prepare_corpus()

    token_ids = corpus["token_ids"]

    inputs, targets = create_input_target_pairs(
        token_ids=token_ids,
        block_size=block_size,
    )

    return {
        "records": corpus["records"],
        "tokenizer": corpus["tokenizer"],
        "inputs": inputs,
        "targets": targets,
        "block_size": block_size,
    }


def main() -> None:
    training_data = prepare_training_data()

    inputs = training_data["inputs"]
    targets = training_data["targets"]

    print("CoastMind training-data preparation successful")
    print(f"Number of records: {len(training_data['records'])}")
    print(f"Block size: {training_data['block_size']}")
    print(f"Number of training examples: {inputs.shape[0]}")
    print(f"Tokens per example: {inputs.shape[1]}")
    print(f"Input tensor shape: {tuple(inputs.shape)}")
    print(f"Target tensor shape: {tuple(targets.shape)}")
    print(f"First input: {inputs[0].tolist()}")
    print(f"First target: {targets[0].tolist()}")


if __name__ == "__main__":
    main()
