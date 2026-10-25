# ECG Heartbeat MLOps

[![CI](https://github.com/marcmaldonadolorca/ecg-heartbeat-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/marcmaldonadolorca/ecg-heartbeat-mlops/actions/workflows/ci.yml)

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
accuracy on its own is misleading: always predicting "normal" already scores 0.83 accuracy with
0.18 F1 macro. Experiment detail and curves in [`docs/wandb_report.md`](docs/wandb_report.md);
serialised metrics in [`models/metadata.json`](models/metadata.json) and
[`models/evaluation.json`](models/evaluation.json).

> **The split is intra-patient, so these numbers are optimistic.** The Kaggle CSVs carry no
> record or patient ID. The shipped train/test files are an 80/20 split of the 109,446 beats
> (87,554 / 21,892) with class proportions identical to four decimals in both files: the signature
> of a stratified random split of beats, which a split by record could not match that closely.
> The validation set here is also a stratified random split of beats (`prepare_arrays` in
> [`src/ecg_mlops/data.py`](src/ecg_mlops/data.py)). Beats from the same patients therefore appear
> in train, validation and test. An inter-patient evaluation (e.g. the
> DS1/DS2 record split of de Chazal et al., 2004) is the clinically relevant setting, and
> minority-class scores typically drop sharply there. It is not done in this repository.

### Baselines (same test set, 21,892 beats)

| Model | Accuracy | F1 macro | Source |
| --- | ---: | ---: | --- |
| Majority class (always N) | 0.8276 | 0.1811 | `python -m ecg_mlops.evaluate` |
| Logistic regression on the 187 raw samples | 0.9135 | 0.6418 | `python -m ecg_mlops.evaluate` |
| Random forest on 16 handcrafted features | 0.9253 | 0.7407 | [`notebook/main.ipynb`](notebook/main.ipynb) |
| MLP (PyTorch) | 0.9607 | 0.7911 | [`notebook/main.ipynb`](notebook/main.ipynb) |
| BiLSTM (PyTorch) | 0.9345 | 0.5590 | [`notebook/main.ipynb`](notebook/main.ipynb) |
| **1D CNN (deployed model)** | **0.9784** | **0.8766** | `python -m ecg_mlops.evaluate` |

The notebook rows come from the original notebook run, with the same seed and the same
train/validation/test split.

### Per-class metrics of the deployed CNN

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| N — Normal | 0.983 | 0.995 | 0.989 | 18,118 |
| S — Supraventricular | 0.933 | 0.601 | 0.731 | 556 |
| V — Ventricular | 0.944 | 0.944 | 0.944 | 1,448 |
| F — Fusion | 0.757 | 0.710 | 0.732 | 162 |
| Q — Unclassifiable | 0.992 | 0.981 | 0.987 | 1,608 |

Confusion matrix (rows: true class, columns: predicted class):

| true \ pred | N | S | V | F | Q |
| --- | ---: | ---: | ---: | ---: | ---: |
| **N** | 18,026 | 22 | 51 | 9 | 10 |
| **S** | 207 | 334 | 10 | 5 | 0 |
| **V** | 55 | 1 | 1,367 | 23 | 2 |
| **F** | 31 | 0 | 16 | 115 | 0 |
| **Q** | 25 | 1 | 4 | 0 | 1,578 |

The weak spot is the supraventricular class: 207 of its 556 beats (37%) are classified as normal,
which caps its recall at 0.60. The logistic regression baseline misses most ventricular beats
(recall 0.33 against 0.94 for the CNN): that is where the convolutional features earn their keep.

To regenerate all of the above from the saved artefact (downloads the public dataset with KaggleHub
on first run; under two minutes on CPU):

```bash
export PYTHONPATH=src
python -m ecg_mlops.evaluate      # writes models/evaluation.json
```

The original notebook is kept in `notebook/main.ipynb`. Around it the project adds a training
script, an API, tests, Docker, W&B tracking and a GitHub Actions workflow.

## Layout

```text
config/                  Hyperparameters and paths
models/                  Trained model, metadata and evaluation.json
notebook/                Original notebook the project grew from
docs/                    W&B report text
scripts/                 Utilities for pushing results to W&B
src/ecg_mlops/           Data, model, training, evaluation and API code
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

The same API is deployed on a free Render instance. **It sleeps when idle: the first request can
take 30–60 s while it wakes up**, then answers immediately.

```bash
curl -s https://ecg-heartbeat-mlops.onrender.com/health
# {"status":"ok","model_loaded":true,"model_path":"/app/models/ecg_cnn.pt"}

# Classify the first beat of the MIT-BIH test file (the evaluate or train command downloads it)
python -c "import csv, json; row = next(csv.reader(open('data/heartbeat/mitbih_test.csv'))); print(json.dumps({'signal': [float(v) for v in row[:187]]}))" \
  | curl -s -X POST https://ecg-heartbeat-mlops.onrender.com/predict \
      -H 'Content-Type: application/json' -d @-
# {"prediction":0,"label":"N - Normal","probability":0.998..., "probabilities":{...}}
```

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

- **Intra-patient split** (see Results): beats of the same patient are in train and test, so the
  scores overstate how the model would do on a new patient. An inter-patient evaluation would
  need the raw MIT-BIH records, which the preprocessed Kaggle CSVs do not identify.
- The five classes come from the preprocessed MIT-BIH dataset, not from raw signals: beat
  segmentation and resampling are inherited from it, not reimplemented here.
- A single train/test split, as the dataset ships it; no cross-validation and no confidence
  intervals.
- The supraventricular class is the weakest (recall 0.60): most of its misses are predicted as
  normal.

## Links

- [Weights & Biases run](https://wandb.ai/maldonadolorcamarc-real/ecg-heartbeat-mlops/runs/ta17nn2d)
  · [W&B report](https://wandb.ai/maldonadolorcamarc-real/ecg-heartbeat-mlops/reports/Analisis-MLOps---Clasificacion-de-latidos-ECG--VmlldzoxNjg1OTM0Mg==)
- [Deployed endpoint](https://ecg-heartbeat-mlops.onrender.com) — free Render instance: the first
  request can take 30–60 s while it wakes up.
