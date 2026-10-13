
# CoastMind Training and Checkpoints

## 1. Purpose

This document explains how CoastMind is trained, how training checkpoints are saved, and how a trained model can be reloaded for inference.

CoastMind currently uses a small, synthetic pilot corpus to test its language-model training pipeline. Training results from this corpus should not be interpreted as evidence of scientific predictive performance.

## 2. Training Configuration

Training parameters are defined in `configs/training_config.py`.

The current configuration includes:

- Random seed: 42
- Block size: 32 tokens
- Embedding dimension: 32
- Attention heads: 4
- Feed-forward hidden dimension: 64
- Transformer blocks: 2
- Batch size: 16
- Learning rate: 0.001
- Training epochs: 10
- Device: CPU

## 3. Running Training

Activate the project virtual environment and run the training module from the project root:

```powershell
python -m src.training.trainer
```

The training pipeline prepares the corpus, creates the tokenizer and model, constructs the data loader, and optimizes the model using AdamW and cross-entropy loss.

After training completes, the model and tokenizer are saved to:

`outputs/coastmind_checkpoint.pt`

The output directory is created automatically if it does not exist.

A different checkpoint location can be supplied by calling `train(checkpoint_path=...)` from Python.

## 4. What a Checkpoint Contains

The checkpoint stores:

- `checkpoint_version`: identifies the checkpoint format.
- `model_config`: architecture settings required to reconstruct the model.
- `model_state_dict`: learned model parameters.
- `vocabulary`: the ordered character vocabulary used by the tokenizer.

The checkpoint does not currently store optimizer state or the current training epoch. Therefore, it supports model reuse and inference, but not exact resumption of interrupted training.

## 5. Loading a Checkpoint

Use the checkpoint loader:

```python
from src.training.checkpoint import load_checkpoint

model, tokenizer = load_checkpoint(
    "outputs/coastmind_checkpoint.pt"
)
```

The loader reconstructs the model, restores its parameters, rebuilds the tokenizer, and switches the model to evaluation mode.

The current implementation loads checkpoints onto the CPU.

## 6. Generating Predictions

The loaded model returns logits for each input token position. These logits can later be converted into token probabilities and used in an autoregressive text-generation loop.

A dedicated text-generation utility has not yet been implemented.

## 7. Testing

Run the complete test suite from the project root:

```powershell
python -m unittest discover -s tests -v
```

Checkpoint tests verify that:

- Checkpoint files are created.
- Models and tokenizers can be reloaded.
- Reloaded models reproduce the original logits for identical inputs.
- Tokenizer encoding and decoding remain consistent.
- Unsupported checkpoint versions and invalid checkpoint data are rejected.

## 8. Limitations and Next Steps

The current pilot corpus is small and synthetic. The model has not yet been validated on independent real-world coastal observations.

Future work should include:

1. Building a documented, licensed, real-world coastal science corpus.
2. Implementing autoregressive text generation.
3. Adding validation data and meaningful evaluation metrics.
4. Supporting optimizer-state checkpoints for resumable training.
5. Developing separate numerical models for coastal processes such as shoreline change and sediment transport.
