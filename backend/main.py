
"""FastAPI backend for CoastMind."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from src.training.checkpoint import load_checkpoint
from src.interpretability.predictions import analyze_next_token
from src.interpretability.attention import get_attention_weights

CHECKPOINT_PATH = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "coastmind_checkpoint.pt"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the trained model once when the API starts."""
    if not CHECKPOINT_PATH.exists():
        raise RuntimeError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    model, tokenizer = load_checkpoint(str(CHECKPOINT_PATH))

    app.state.model = model
    app.state.tokenizer = tokenizer

    yield


app = FastAPI(
    title="CoastMind API",
    description=(
        "API for CoastMind language-model predictions "
        "and interpretability."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Report whether the API service is running."""
    return {
        "status": "healthy",
        "service": "coastmind-api",
    }


@app.get("/model-info")
def model_info() -> dict:
    """Return basic information about the loaded model."""
    model = getattr(app.state, "model", None)

    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded.",
        )

    return {
        "model_name": "CoastMind",
        "vocabulary_size": len(app.state.tokenizer.itos),
        "max_sequence_length": model.max_seq_length,
        "number_of_layers": len(model.transformer_blocks),
        "number_of_attention_heads": (
            model.transformer_blocks[0].attention.num_heads
        ),
        "embedding_dimension": (
            model.embedding.token_embedding.embedding.embedding_dim
        ),
        "device": str(next(model.parameters()).device),
    }


from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=32)
    temperature: float = Field(default=1.0, gt=0)


class AttentionRequest(BaseModel):
    prompt: str = Field(min_length=1)
    block_index: int = Field(default=0, ge=0)



@app.post("/predict")
def predict_next_token(request: PredictionRequest) -> dict:
    """Predict likely next characters for a supplied prompt."""
    model = app.state.model
    tokenizer = app.state.tokenizer

    try:
        result = analyze_next_token(
            model=model,
            tokenizer=tokenizer,
            prompt=request.prompt,
            top_k=request.top_k,
            temperature=request.temperature,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    return {
        "prompt": request.prompt,
        "top_k": request.top_k,
        "temperature": request.temperature,
        "predictions": result,
    }


@app.post("/attention")
def inspect_attention(request: AttentionRequest) -> dict:
    """Return Transformer attention weights for a text prompt."""
    model = app.state.model
    tokenizer = app.state.tokenizer

    try:
        token_ids = tokenizer.encode(request.prompt)

        if len(token_ids) > model.max_seq_length:
            raise ValueError(
                f"Prompt is too long. Maximum length is "
                f"{model.max_seq_length} characters."
            )

        if request.block_index >= len(model.transformer_blocks):
            raise ValueError(
                f"Invalid block_index. Choose a value from "
                f"0 to {len(model.transformer_blocks) - 1}."
            )

        attention_weights = get_attention_weights(
            model=model,
            token_ids=__import__("torch").tensor(
                [token_ids],
                dtype=__import__("torch").long,
            ),
            block_index=request.block_index,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    attention_matrix = attention_weights[0].tolist()

    return {
        "prompt": request.prompt,
        "block_index": request.block_index,
        "sequence_length": len(token_ids),
        "number_of_heads": len(attention_matrix),
        "attention_weights": attention_matrix,
    }
