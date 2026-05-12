from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ecg_mlops.config import resolve_project_path
from ecg_mlops.data import CLASS_NAMES, N_TIMESTEPS
from ecg_mlops.predict import load_checkpoint, predict_signal


class ECGInput(BaseModel):
    signal: list[float] = Field(
        ...,
        min_length=N_TIMESTEPS,
        max_length=N_TIMESTEPS,
        description="Latido ECG preprocesado con 187 puntos temporales.",
    )


def _configured_model_path() -> Path:
    return resolve_project_path(os.getenv("MODEL_PATH", "models/ecg_cnn.pt"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = _configured_model_path()
    app.state.model = None
    app.state.model_path = str(model_path)
    app.state.class_names = CLASS_NAMES

    if model_path.exists():
        model, checkpoint = load_checkpoint(model_path)
        app.state.model = model
        app.state.checkpoint = checkpoint

    yield


app = FastAPI(
    title="ECG Heartbeat MLOps API",
    description="API de inferencia para clasificar latidos ECG MIT-BIH.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    return {
        "service": "ecg-heartbeat-mlops",
        "message": "API de clasificacion de latidos ECG.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.model is not None,
        "model_path": app.state.model_path,
    }


@app.post("/predict")
def predict(payload: ECGInput):
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact not found. Train the model or provide MODEL_PATH.",
        )
    return predict_signal(app.state.model, payload.signal)

