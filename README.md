# CoastMind

**An Interpretable Coastal Intelligence Language Model**

## Overview

CoastMind is an educational AI research project focused on building a small decoder-only Transformer language model for coastal science.

The project aims to make the internal computations of a language model inspectable, from tokenization and embeddings to self-attention, contextual representations, and next-token prediction.

## Objectives

- Build a tokenizer for coastal-science text.
- Understand and implement token embeddings and positional information.
- Implement query, key, and value projections and self-attention.
- Explore causal masking, softmax, and context vectors.
- Build Transformer blocks and a language-model head.
- Train and evaluate a small language model.
- Develop an interactive laboratory for inspecting model computations.

## Planned Project Structure

- `src/` — model, data-processing, and training code.
- `data/` — raw and processed datasets.
- `configs/` — model and experiment settings.
- `experiments/` — research notebooks and controlled experiments.
- `app/` — interactive visualisation interface.
- `tests/` — automated tests.
- `outputs/` — experiment results and model outputs.
- `docs/` — technical documentation.

## Technology Stack

- Python
- PyTorch
- NumPy
- Additional libraries will be introduced as the project develops.

## Current Status

**Environment setup completed.**

Python, PyTorch, and NumPy have been installed and verified in the project's virtual environment. The initial folder structure has been created.

Tokenizer implementation, Transformer development, model training, and the interactive interface are planned next.

## Learning Philosophy

CoastMind is being developed incrementally. Each component will be explained, implemented, tested, and documented before the project advances to the next stage.