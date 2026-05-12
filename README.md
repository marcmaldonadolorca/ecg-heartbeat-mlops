# ECG Heartbeat MLOps

Autor: Marc Maldonado

Proyecto MLOps para clasificar latidos ECG del dataset MIT-BIH Arrhythmia,
incluido en ECG Heartbeat Categorization Dataset. El modelo principal es una
CNN 1D en PyTorch que recibe un latido preprocesado de 187 puntos temporales y
devuelve una de cinco clases:

| Clase | Etiqueta |
| --- | --- |
| 0 | N - Normal |
| 1 | S - Supraventricular |
| 2 | V - Ventricular |
| 3 | F - Fusion |
| 4 | Q - No clasificable |

El notebook original se conserva en `notebook/main.ipynb`. El codigo de
produccion esta separado en `src/`, con entrenamiento reproducible,
infererencia, API, tests, Docker y CI.

## Estructura

```text
config/                  Hiperparametros y rutas
models/                  Artefactos del modelo entrenado
notebook/                Notebook original del proyecto de la asignatura previa
docs/                    Texto de entrega y borrador del report W&B
scripts/                 Utilidades para subir resultados a W&B
src/ecg_mlops/           Codigo de datos, modelo, entrenamiento e API
tests/                   Tests de datos, modelo y API
.github/workflows/ci.yml Pipeline de integracion continua
Dockerfile               Imagen para servir la API
render.yaml              Despliegue como Web Service en Render
```

## Entorno local

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

En Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Entrenamiento

El dataset se descarga automaticamente con KaggleHub la primera vez:

```bash
export PYTHONPATH=src
python -m ecg_mlops.train --config config/config.yaml
```

Para una prueba rapida:

```bash
export PYTHONPATH=src
python -m ecg_mlops.train --epochs 1 --max-train-samples 1000 --max-test-samples 300
```

En Windows PowerShell se puede usar:

```powershell
$env:PYTHONPATH="src"
python -m ecg_mlops.train --config config/config.yaml
```

Para registrar el experimento en Weights & Biases:

```bash
export PYTHONPATH=src
wandb login
python -m ecg_mlops.train --config config/config.yaml --use-wandb
```

Si el modelo ya esta entrenado, tambien se puede subir el artefacto y el
historial guardado con:

```bash
python scripts/log_existing_model_to_wandb.py --project ecg-heartbeat-mlops
```

El texto base del report esta en `docs/wandb_report.md`.

El entrenamiento guarda el artefacto en `models/ecg_cnn.pt` y metadatos en
`models/metadata.json`.

Resultado obtenido con la configuracion por defecto:

| Metrica | Valor |
| --- | ---: |
| Accuracy test | 0.9784 |
| F1 macro test | 0.8766 |
| F1 weighted test | 0.9772 |

## API local

```bash
export PYTHONPATH=src
uvicorn ecg_mlops.api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints principales:

- `GET /health`: estado del servicio y carga del modelo.
- `POST /predict`: clasificacion de un latido ECG.

La forma mas comoda de probar la API en local es abrir
`http://127.0.0.1:8000/docs`. El endpoint `/predict` espera un JSON con un
campo `signal` que contenga exactamente 187 valores numericos.

## Tests

```bash
pytest -q
```

Los tests cubren validacion de datos, shapes del modelo, gradientes y API con
`TestClient`.

## Docker

```bash
docker build -t ecg-heartbeat-mlops .
docker run --rm -p 8000:8000 -e PORT=8000 ecg-heartbeat-mlops
```

## CI/CD

El workflow `.github/workflows/ci.yml` ejecuta:

1. Instalacion de dependencias.
2. Tests con `pytest`.
3. Build de la imagen Docker.

## Despliegue

La configuracion `render.yaml` permite desplegar la API como Web Service Docker
en Render. El servicio expone un endpoint publico `onrender.com` y usa
`/health` como health check.

## Enlaces de entrega

- GitHub: https://github.com/marcmaldonadolorca/ecg-heartbeat-mlops
- Weights & Biases: pendiente de publicar el report
- Endpoint en produccion: pendiente de desplegar
