from __future__ import annotations

import argparse
import copy
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from ecg_mlops.config import load_config, resolve_project_path
from ecg_mlops.data import CLASS_NAMES, ECGDataset, load_raw_data, prepare_arrays
from ecg_mlops.model import ECGCNN


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, Any]:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    y_true: list[int] = []
    y_pred: list[int] = []

    for xb, yb in dataloader:
        xb = xb.to(device)
        yb = yb.to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(xb)
            loss = criterion(logits, yb)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * xb.size(0)
        preds = logits.argmax(dim=1)
        y_true.extend(yb.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    return {
        "loss": total_loss / len(dataloader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }


def _init_wandb(cfg: dict[str, Any], use_wandb: bool):
    if not use_wandb:
        return None

    import wandb

    wandb_cfg = cfg.get("wandb", {})
    return wandb.init(
        project=wandb_cfg.get("project", "ecg-heartbeat-mlops"),
        entity=wandb_cfg.get("entity"),
        job_type=wandb_cfg.get("job_type", "training"),
        config=cfg,
    )


def train(cfg: dict[str, Any], use_wandb: bool = False) -> dict[str, Any]:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    seed = int(cfg["train"]["seed"])
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    data_cfg = cfg["data"]
    df_train, df_test = load_raw_data(
        data_cfg["dir"],
        dataset_name=data_cfg["dataset"],
        download=bool(data_cfg.get("download", True)),
    )
    arrays = prepare_arrays(
        df_train,
        df_test,
        validation_size=float(data_cfg["validation_size"]),
        seed=seed,
        max_train_samples=data_cfg.get("max_train_samples"),
        max_test_samples=data_cfg.get("max_test_samples"),
    )

    batch_size = int(data_cfg["batch_size"])
    train_dl = DataLoader(
        ECGDataset(arrays.X_train, arrays.y_train),
        batch_size=batch_size,
        shuffle=True,
    )
    val_dl = DataLoader(
        ECGDataset(arrays.X_val, arrays.y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_dl = DataLoader(
        ECGDataset(arrays.X_test, arrays.y_test),
        batch_size=batch_size,
        shuffle=False,
    )

    model_cfg = cfg["model"]
    model = ECGCNN(
        n_classes=int(model_cfg["n_classes"]),
        base_filters=int(model_cfg["base_filters"]),
        dropout=float(model_cfg["dropout"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["train"]["learning_rate"]))
    wandb_run = _init_wandb(cfg, use_wandb)

    best_state = copy.deepcopy(model.state_dict())
    best_val_f1 = -1.0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        train_metrics = run_epoch(model, train_dl, criterion, device, optimizer)
        val_metrics = run_epoch(model, val_dl, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
        }
        history.append(row)

        if wandb_run:
            import wandb

            wandb.log(row)

        logger.info(
            "Epoch %s | train loss %.4f f1 %.4f | val loss %.4f f1 %.4f",
            epoch,
            row["train_loss"],
            row["train_f1_macro"],
            row["val_loss"],
            row["val_f1_macro"],
        )

        if row["val_f1_macro"] > best_val_f1:
            best_val_f1 = row["val_f1_macro"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    test_metrics = run_epoch(model, test_dl, criterion, device)
    logger.info("Test metrics: %s", test_metrics)

    artifact_cfg = cfg["artifacts"]
    model_dir = resolve_project_path(artifact_cfg["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / artifact_cfg["model_name"]
    metadata_path = model_dir / "metadata.json"

    checkpoint = {
        "model_state_dict": {key: value.cpu() for key, value in best_state.items()},
        "model_config": model_cfg,
        "class_names": CLASS_NAMES,
        "history": history,
        "test_metrics": test_metrics,
        "config": cfg,
    }
    torch.save(checkpoint, model_path)

    metadata = {
        "model_path": str(Path(artifact_cfg["model_dir"]) / artifact_cfg["model_name"]),
        "best_val_f1_macro": best_val_f1,
        "test_metrics": test_metrics,
        "class_names": CLASS_NAMES,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if wandb_run:
        import wandb

        wandb.log({f"test/{key}": value for key, value in test_metrics.items()})
        artifact = wandb.Artifact(
            name="ecg-cnn-model",
            type="model",
            description="CNN 1D entrenada para clasificacion de latidos ECG MIT-BIH.",
        )
        artifact.add_file(str(model_path))
        artifact.add_file(str(metadata_path))
        wandb.log_artifact(artifact)
        wandb_run.finish()

    return {
        "model_path": model_path,
        "metadata_path": metadata_path,
        "best_val_f1_macro": best_val_f1,
        "test_metrics": test_metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ECG heartbeat CNN model.")
    parser.add_argument("--config", default="config/config.yaml", help="Path to YAML config.")
    parser.add_argument("--use-wandb", action="store_true", help="Log training to Weights & Biases.")
    parser.add_argument("--epochs", type=int, default=None, help="Override number of epochs.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Limit train rows.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Limit test rows.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg["train"]["epochs"] = args.epochs
    if args.max_train_samples is not None:
        cfg["data"]["max_train_samples"] = args.max_train_samples
    if args.max_test_samples is not None:
        cfg["data"]["max_test_samples"] = args.max_test_samples

    result = train(cfg, use_wandb=args.use_wandb)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
