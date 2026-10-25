"""Evaluate the saved CNN on the test split and compare it with two baselines.

Writes per-class metrics, the confusion matrix and the baselines to
``models/evaluation.json``. Runs on CPU in about a minute:

    python -m ecg_mlops.evaluate
"""

from __future__ import annotations

import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score

from ecg_mlops.config import load_config, resolve_project_path
from ecg_mlops.data import CLASS_NAMES, load_raw_data, prepare_arrays
from ecg_mlops.predict import load_checkpoint


def scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    labels = sorted(CLASS_NAMES)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    return {
        "accuracy": report["accuracy"],
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "per_class": {CLASS_NAMES[c]: report[str(c)] for c in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def main() -> None:
    cfg = load_config()
    a = prepare_arrays(
        *load_raw_data(cfg["data"]["dir"], cfg["data"]["dataset"]),
        validation_size=float(cfg["data"]["validation_size"]),
        seed=int(cfg["train"]["seed"]),
    )

    model, _ = load_checkpoint(resolve_project_path("models/ecg_cnn.pt"))
    with torch.no_grad():
        logits = torch.cat([model(x) for x in torch.from_numpy(a.X_test).unsqueeze(1).split(2048)])
    cnn_pred = logits.argmax(dim=1).numpy()

    majority = np.bincount(a.y_train).argmax()
    logreg = LogisticRegression(max_iter=2000).fit(a.X_train, a.y_train)

    results = {
        "test_rows": len(a.y_test),
        "cnn": scores(a.y_test, cnn_pred),
        "majority_class": scores(a.y_test, np.full_like(a.y_test, majority)),
        "logistic_regression": scores(a.y_test, logreg.predict(a.X_test)),
    }
    out = resolve_project_path("models/evaluation.json")
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    for name in ("majority_class", "logistic_regression", "cnn"):
        print(f"{name:20s} accuracy {results[name]['accuracy']:.4f}  f1_macro {results[name]['f1_macro']:.4f}")
    print(f"Written {out}")


if __name__ == "__main__":
    main()
