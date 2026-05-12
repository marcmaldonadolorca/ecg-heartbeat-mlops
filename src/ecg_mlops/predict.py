from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from ecg_mlops.data import CLASS_NAMES, N_TIMESTEPS
from ecg_mlops.model import ECGCNN


def load_checkpoint(model_path: str | Path) -> tuple[ECGCNN, dict[str, Any]]:
    checkpoint = torch.load(Path(model_path), map_location="cpu", weights_only=False)
    model_config = checkpoint.get("model_config", {})
    model = ECGCNN(
        n_classes=int(model_config.get("n_classes", len(CLASS_NAMES))),
        base_filters=int(model_config.get("base_filters", 32)),
        dropout=float(model_config.get("dropout", 0.10)),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_signal(model: ECGCNN, signal: list[float] | np.ndarray) -> dict[str, Any]:
    signal_array = np.asarray(signal, dtype=np.float32)
    if signal_array.shape != (N_TIMESTEPS,):
        raise ValueError(f"Expected signal with shape ({N_TIMESTEPS},), got {signal_array.shape}.")

    x = torch.tensor(signal_array, dtype=torch.float32).view(1, 1, N_TIMESTEPS)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    prediction = int(probabilities.argmax())
    return {
        "prediction": prediction,
        "label": CLASS_NAMES[prediction],
        "probability": float(probabilities[prediction]),
        "probabilities": {
            str(class_id): float(probabilities[class_id]) for class_id in sorted(CLASS_NAMES)
        },
    }

