
"""CoastMind Explorer: an interactive model interpretability app."""

import torch

from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Add the project root so the app can import existing CoastMind modules.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.interpretability.predictions import analyze_next_token
from src.interpretability.attention import get_attention_weights
from src.training.checkpoint import load_checkpoint


CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "coastmind_checkpoint.pt"


@st.cache_resource
def get_model():
    """Load the trained model and tokenizer once per app session."""
    return load_checkpoint(CHECKPOINT_PATH)


st.set_page_config(
    page_title="CoastMind Explorer",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 CoastMind Explorer")
st.caption(
    "Explore how a small GPT-style Transformer predicts the next character."
)

st.info(
    "Educational prototype: the current model uses a tiny synthetic corpus. "
    "Its predictions are for software exploration, not scientific coastal "
    "forecasting."
)

if not CHECKPOINT_PATH.is_file():
    st.error(f"Model checkpoint not found: {CHECKPOINT_PATH}")
    st.stop()

try:
    model, tokenizer = get_model()
except Exception as exc:
    st.error(f"Could not load the CoastMind checkpoint: {exc}")
    st.stop()

st.sidebar.header("Prediction settings")

prompt = st.sidebar.text_input(
    "Input prompt",
    value="coastal",
    help="Enter text made up of characters in the model's vocabulary.",
)

top_k = st.sidebar.slider(
    "Number of candidate characters (top-k)",
    min_value=1,
    max_value=model.vocab_size,
    value=min(5, model.vocab_size),
)

temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help=(
        "Lower values make the probability distribution more concentrated; "
        "higher values make it more spread out."
    ),
)

st.subheader("Next-character predictions")

if st.button("Analyze prompt", type="primary"):
    if not prompt:
        st.warning("Enter a prompt before running the analysis.")
    else:
        try:
            predictions = analyze_next_token(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                top_k=top_k,
                temperature=temperature,
            )

            dataframe = pd.DataFrame(predictions)
            dataframe["display_token"] = dataframe["token"].map(
                {
                    " ": "SPACE",
                    "\t": "TAB",
                    "\n": "NEWLINE",
                    "\r": "CARRIAGE RETURN",
                }
            ).fillna(dataframe["token"].map(repr))

            dataframe["probability_percent"] = (
                dataframe["probability"] * 100
            )

            left, right = st.columns(2)

            with left:
                st.metric("Characters in prompt", len(prompt))
                st.metric("Candidates displayed", len(dataframe))

            with right:
                st.metric("Temperature", f"{temperature:.1f}")
                st.metric("Vocabulary size", model.vocab_size)

            chart_data = dataframe.set_index("display_token")[
                "probability"
            ]

            st.bar_chart(chart_data)

            st.markdown("#### Prediction details")

            display_table = dataframe[
                [
                    "display_token",
                    "token_id",
                    "logit",
                    "probability_percent",
                ]
            ].rename(
                columns={
                    "display_token": "Candidate character",
                    "token_id": "Token ID",
                    "logit": "Raw logit",
                    "probability_percent": "Probability (%)",
                }
            )

            st.dataframe(
                display_table,
                hide_index=True,
                use_container_width=True,
            )

        except (ValueError, IndexError) as exc:
            st.warning(f"Unable to analyze this prompt: {exc}")

st.divider()
st.caption(
    "CoastMind is a learning and interpretability laboratory. "
    "Coastal-science prediction capabilities require separate real-world "
    "data, scientific modelling, and validation."
)

st.divider()

st.header("Attention Explorer")
st.write(
    "Inspect how each character attends to earlier characters "
    "in the input sequence."
)

attention_prompt = st.text_input(
    "Prompt for attention analysis",
    value="coastal",
    key="attention_prompt",
)

block_index = st.selectbox(
    "Transformer block",
    options=list(range(len(model.transformer_blocks))),
    format_func=lambda index: f"Block {index}",
)

head_index = st.selectbox(
    "Attention head",
    options=list(
        range(model.transformer_blocks[block_index].attention.num_heads)
    ),
    format_func=lambda index: f"Head {index}",
)

if st.button("Analyze attention", type="primary"):
    if not attention_prompt:
        st.warning("Enter a prompt before analyzing attention.")
    else:
        try:
            attention_token_ids = tokenizer.encode(attention_prompt)

            attention_tensor = torch.tensor(
                [attention_token_ids],
                dtype=torch.long,
            )

            attention_weights = get_attention_weights(
                model=model,
                token_ids=attention_tensor,
                block_index=block_index,
            )

            attention_matrix = (
                attention_weights[0, head_index].cpu().numpy()
            )

            characters = [
                {
                    "Position": position,
                    "Character": {
                        " ": "SPACE",
                        "\t": "TAB",
                        "\n": "NEWLINE",
                        "\r": "CARRIAGE RETURN",
                    }.get(character, repr(character)),
                }
                for position, character in enumerate(attention_prompt)
            ]

            attention_frame = pd.DataFrame(
                attention_matrix,
                columns=[
                    item["Character"] for item in characters
                ],
                index=[
                    f'{item["Position"]}: {item["Character"]}'
                    for item in characters
                ],
            )

            st.subheader(
                f"Block {block_index} · Head {head_index}"
            )

            st.caption(
                "Rows are query positions; columns are key positions. "
                "Future positions should have zero attention."
            )

            st.dataframe(
                attention_frame.style.format("{:.3f}"),
                use_container_width=True,
            )

            st.markdown("#### Attention heatmap")

            st.write(
                "Darker and lighter cells indicate different attention "
                "weights. Each row represents a probability distribution."
            )

            st.dataframe(
                attention_frame.style.background_gradient(
                    cmap="viridis",
                    vmin=0,
                    vmax=1,
                ).format("{:.3f}"),
                use_container_width=True,
            )

        except (ValueError, IndexError) as exc:
            st.warning(f"Unable to analyze attention: {exc}")
