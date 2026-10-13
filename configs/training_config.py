
"""Configuration for CoastMind's first language-model training experiment."""

# Reproducibility
RANDOM_SEED = 42

# Dataset and sequence preparation
BLOCK_SIZE = 32

# Model architecture
EMBEDDING_DIM = 32
NUM_HEADS = 4
HIDDEN_DIM = 64
NUM_LAYERS = 2

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 0.001
NUM_EPOCHS = 10

# Runtime
DEVICE = "cpu"
