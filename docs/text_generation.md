# CoastMind Text Generation

## 1. Overview

CoastMind uses autoregressive text generation to produce text one character at a time. It uses a trained decoder-only Transformer language model to predict the next character based on the characters that came before it.

The implementation is located in:

- `src/generation/generate.py`

The generation process uses:

- A trained `CoastMindLanguageModel`
- A `CharacterTokenizer`
- A starting prompt
- A maximum number of new characters
- A temperature parameter to control sampling randomness

## 2. How Text Generation Works

The `generate_text()` function performs the following steps:

1. Validates the prompt, token count, and temperature.
2. Encodes the prompt into token IDs.
3. Passes the current sequence through the Transformer.
4. Extracts the logits for the final sequence position.
5. Divides the logits by the temperature.
6. Converts the logits into probabilities using softmax.
7. Samples the next character using `torch.multinomial()`.
8. Appends the sampled character to the sequence.
9. Repeats the process until the requested number of new characters has been generated.
10. Decodes the complete sequence into text.

## 3. Understanding the Important Parameters

### Prompt

The prompt is the initial text from which generation begins.

Example:

`coastal`

The model preserves this prompt and generates additional characters after it.

### Maximum New Tokens

The `max_new_tokens` parameter determines how many additional characters the model generates.

For example, if the prompt contains seven characters and `max_new_tokens=5`, the returned string contains twelve characters, assuming every character is represented by one token.

### Temperature

Temperature controls how the model's next-character probabilities are distributed before sampling.

- Lower temperatures make the distribution more concentrated around higher-probability characters.
- Higher temperatures make the distribution flatter and can increase randomness.
- Temperature must be greater than zero.

Temperature does not give the model new knowledge or guarantee coherent output.

### Context Window

The model can process only a limited number of characters at once. CoastMind uses the most recent characters up to `model.max_seq_length` when predicting the next character.

The complete generated sequence is retained for the final decoded output, even when older characters fall outside the model's active context window.

## 4. Loading a Checkpoint and Generating Text

After training a model, CoastMind saves a checkpoint containing the model configuration, trained weights, and vocabulary.

The following command loads that checkpoint and generates text:

```powershell
python -c "from src.training.checkpoint import load_checkpoint; from src.generation.generate import generate_text; model, tokenizer = load_checkpoint('outputs/coastmind_checkpoint.pt'); print(generate_text(model, tokenizer, prompt='coastal', max_new_tokens=80, temperature=0.8))"
```

The checkpoint is stored at:

`outputs/coastmind_checkpoint.pt`

The checkpoint is a generated training artifact and should not be committed to Git.

## 5. Testing

The generation tests are located in:

`tests/test_generation.py`

Run them using:

```powershell
python -m unittest discover -s tests -p "test_generation.py" -v
```

The tests cover:

- Returning a string
- Preserving the prompt
- Generating the requested number of characters
- Returning the prompt when zero new characters are requested
- Rejecting an empty prompt
- Rejecting a negative token count
- Rejecting a zero temperature
- Switching the model to evaluation mode
- Generating text after checkpoint reload

## 6. Current Limitations

The current CoastMind pilot corpus contains only three synthetic records. Its vocabulary is character-level, and the model is small.

Consequently:

- Generated text may be incoherent or repetitive.
- A decreasing training loss does not establish scientific accuracy.
- The model has not been validated on unseen coastal-science text.
- Generated text must not be treated as a scientific prediction or measurement.

Future improvements should include a larger, carefully reviewed coastal-science corpus, evaluation on held-out data, and experiments with alternative tokenization and model configurations.

Text generation is one component of CoastMind. Predicting shoreline change, sediment transport, and coastal morphodynamics will require separate numerical modelling, relevant environmental observations, scientific validation, and uncertainty assessment.
