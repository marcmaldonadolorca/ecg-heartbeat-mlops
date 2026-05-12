import torch
from fastapi.testclient import TestClient

from ecg_mlops.api import app
from ecg_mlops.data import N_TIMESTEPS
from ecg_mlops.model import ECGCNN


def test_api_health_and_prediction(tmp_path, monkeypatch):
    model = ECGCNN(n_classes=5, base_filters=8, dropout=0.1)
    model_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": {"n_classes": 5, "base_filters": 8, "dropout": 0.1},
        },
        model_path,
    )
    monkeypatch.setenv("MODEL_PATH", str(model_path))

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["model_loaded"] is True

        response = client.post("/predict", json={"signal": [0.0] * N_TIMESTEPS})
        assert response.status_code == 200
        payload = response.json()
        assert set(payload) == {"prediction", "label", "probability", "probabilities"}
        assert len(payload["probabilities"]) == 5

