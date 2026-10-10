
# CoastMind: Attention Mechanism

## 1. Overview

The attention mechanism allows CoastMind to build a contextual representation
of each token by combining information from other token positions.

CoastMind currently implements a single-head, causal self-attention mechanism
using PyTorch.

The implementation is located in:

- `src/model/attention.py`
- `tests/test_attention.py`

The attention components are tested independently and together.

## 2. Components Implemented

### 2.1 QKVProjection

**Purpose:** Generate Query, Key, and Value vectors from the input embeddings.

The component uses three separate trainable linear transformations:

- Query projection
- Key projection
- Value projection

These transformations allow the model to learn different representations
for comparing tokens and combining their information.

### 2.2 AttentionScores

**Purpose:** Calculate the compatibility scores between Queries and Keys.

The operation is:

`QKᵀ`

Each score indicates how strongly a query position matches a key position
before scaling and causal masking.

### 2.3 AttentionScaling

**Purpose:** Scale the attention scores.

The operation is:

`QKᵀ / √dₖ`

Here, `dₖ` is the key dimension. Scaling helps control the magnitude
of attention scores before applying softmax.

### 2.4 CausalMask

**Purpose:** Prevent each token position from attending to future positions.

The mask is a lower-triangular Boolean matrix:

- `True` means attention is allowed.
- `False` means attention is blocked.

Future-position scores are replaced with negative infinity before softmax.
Consequently, those positions receive zero attention weight.

### 2.5 AttentionSoftmax

**Purpose:** Convert masked attention scores into normalized weights.

Softmax operates over the final tensor dimension. For each query position,
the attention weights sum to one.

### 2.6 ValueAggregation

**Purpose:** Combine Value vectors using the attention weights.

The operation is:

`AttentionWeights × V`

The resulting context tensor contains a weighted combination of the Value
vectors available to each token position.

### 2.7 SelfAttention

**Purpose:** Combine the individual attention components into one module.

The forward pass:

1. Projects the input into Queries, Keys, and Values.
2. Calculates attention scores.
3. Scales the scores.
4. Applies the causal mask.
5. Calculates attention weights using softmax.
6. Aggregates the Value vectors.
7. Returns contextual representations.

The optional `return_attention=True` argument returns both the context tensor
and the attention weights for inspection.

## 3. Mathematical Formulation

The implemented operation is:

`Attention(Q, K, V) = softmax(QKᵀ / √dₖ + M)V`

Here, `M` is the additive causal mask:

- Allowed positions receive `0`.
- Future positions receive negative infinity.

This ensures that future-token scores contribute zero attention weight.

## 4. Testing

The attention test suite currently contains 32 tests.

The tests cover:

- Query, Key, and Value projections.
- Attention score calculations.
- Attention scaling.
- Softmax normalization.
- Causal mask dimensions and behavior.
- Zero attention weights for future positions.
- Value aggregation.
- Self-attention output shapes.
- Attention-weight normalization.
- Invalid embedding dimensions.

**Verified milestone:** On 10 October 2026, the command
`python -m unittest tests.test_attention -v` completed successfully
with 32 tests passing.

## 5. Current Limitations

- The implementation uses one attention head.
- The module has not yet been trained on the coastal-science corpus.
- Attention weights can be inspected, but they do not independently
  establish causal explanations for model predictions.
- Passing unit tests verifies the tested software behavior; it does not
  establish scientific accuracy or language-model quality.

## 6. Next Development Stage

The next stage is multi-head self-attention. After that, CoastMind will
progress toward a Transformer block with residual connections, layer
normalization, and a feed-forward network.
