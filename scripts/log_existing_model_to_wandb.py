from __future__ import annotations

import argparse
from pathlib import Path

import torch
import wandb


def main() -> None:
    parser = argparse.ArgumentParser(description="Sube el modelo entrenado a W&B.")
    parser.add_argument("--project", default="ecg-heartbeat-mlops")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--model-path", default="models/ecg_cnn.pt")
    parser.add_argument("--metadata-path", default="models/metadata.json")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    metadata_path = Path(args.metadata_path)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

    run = wandb.init(
        project=args.project,
        entity=args.entity,
        job_type="training",
        config=checkpoint["config"],
        name="cnn-1d-final",
    )

    for row in checkpoint["history"]:
        wandb.log(row)

    wandb.log({f"test/{k}": v for k, v in checkpoint["test_metrics"].items()})

    artifact = wandb.Artifact(
        name="ecg-cnn-model",
        type="model",
        description="CNN 1D entrenada para clasificacion de latidos ECG MIT-BIH.",
    )
    artifact.add_file(str(model_path))
    artifact.add_file(str(metadata_path))
    wandb.log_artifact(artifact)

    print(f"Run URL: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()

