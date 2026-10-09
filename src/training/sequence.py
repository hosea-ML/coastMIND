
import torch


def create_input_target_pairs(
    token_ids: list[int],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sliding input-target pairs for next-token prediction."""

    if block_size <= 0:
        raise ValueError("block_size must be greater than zero.")

    if len(token_ids) <= block_size:
        raise ValueError(
            "token_ids must contain more tokens than block_size."
        )

    inputs = []
    targets = []

    for start in range(len(token_ids) - block_size):
        input_sequence = token_ids[start : start + block_size]
        target_sequence = token_ids[start + 1 : start + block_size + 1]

        inputs.append(input_sequence)
        targets.append(target_sequence)

    x = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    return x, y
