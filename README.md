# ECG Heartbeat MLOps

Classifying ECG heartbeats from the MIT-BIH Arrhythmia dataset (via the ECG Heartbeat
Categorization Dataset), taken from a notebook to a deployed service. The model is a 1D CNN in
PyTorch that takes a preprocessed beat of 187 time points and returns one of five classes:

| Class | Label |
| --- | --- |
| 0 | N — Normal |
| 1 | S — Supraventricular |
| 2 | V — Ventricular |
| 3 | F — Fusion |
| 4 | Q — Unclassifiable |

## Results (test set, imbalanced)

| Metric | Value |
| --- | --- |
| Accuracy | 0.9784 |
| **F1 macro** | **0.8766** |
| F1 weighted | 0.9772 |
| Best F1 macro (validation) | 0.8932 |

**The selection criterion is F1 macro, not accuracy.** The normal class dominates the dataset, so
accuracy on its own is misleading: a model that never predicts the rare classes still scores above
0.82. The gap between 0.978 accuracy and 0.877 F1 macro is exactly the cost of the minority
classes. Experiment detail and curves in [`docs/wandb_report.md`](docs/wandb_report.md); serialised
metrics in [`models/metadata.json`](models/metadata.json).

The original notebook is kept in `notebook/main.ipynb`. Around it the project adds a training
script, an API, tests, Docker, W&B tracking and a GitHub Actions workflow.

## Layout

```text
config/                  Hyperparameters and paths
models/                  Trained model artefacts
notebook/                Original notebook the project grew from
docs/                    W&B report text
scripts/                 Utilities for pushing results to W&B
src/ecg_mlops/           Data, model, training and API code
tests/                   Data, model and API tests
.github/workflows/ci.yml CI workflow
Dockerfile               Image for serving the API
render.yaml              Deployment as a Render Web Service
```

## Local setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Training

The dataset is downloaded automatically with KaggleHub on first run:

```bash
export PYTHONPATH=src
python -m ecg_mlops.train --config config/config.yaml
```

A quick smoke run:

```bash
python -m ecg_mlops.train --epochs 1 --max-train-samples 1000 --max-test-samples 300
```

To log the experiment to Weights & Biases, `wandb login` and add `--use-wandb`. If the model is
already trained, `python scripts/log_existing_model_to_wandb.py --project ecg-heartbeat-mlops`
uploads the artefact and the saved history. Training writes the artefact to `models/ecg_cnn.pt`
and metadata to `models/metadata.json`.

## API

```bash
export PYTHONPATH=src
uvicorn ecg_mlops.api:app --host 0.0.0.0 --port 8000 --reload
```

- `GET /health` — service state and whether the model is loaded.
- `POST /predict` — classify one ECG beat. Expects JSON with a `signal` field holding exactly 187
  numeric values.

The easiest way to try it locally is `http://127.0.0.1:8000/docs`.

## Tests, Docker and CI

```bash
pytest -q                                      # data validation, model shapes, gradients, API
docker build -t ecg-heartbeat-mlops .
docker run --rm -p 8000:8000 -e PORT=8000 ecg-heartbeat-mlops
```

CI (`.github/workflows/ci.yml`) installs dependencies, runs the tests and builds the Docker image.
`render.yaml` deploys the API as a Docker Web Service on Render, using `/health` as the health
check.

## Limitations

- Per-class metrics are not serialised, so the README cannot show a confusion matrix without
  re-running inference — that is the clearest missing piece.
- The five classes come from the preprocessed MIT-BIH dataset, not from raw signals: beat
  segmentation and resampling are inherited from it, not reimplemented here.
- A single train/test split, as the dataset ships it; no cross-validation.

## Links

- [Weights & Biases run](https://wandb.ai/maldonadolorcamarc-real/ecg-heartbeat-mlops/runs/ta17nn2d)
  · [W&B report](https://wandb.ai/maldonadolorcamarc-real/ecg-heartbeat-mlops/reports/Analisis-MLOps---Clasificacion-de-latidos-ECG--VmlldzoxNjg1OTM0Mg==)
- [Deployed endpoint](https://ecg-heartbeat-mlops.onrender.com)
