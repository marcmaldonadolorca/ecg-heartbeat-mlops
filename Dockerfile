FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/src

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY models/ models/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn ecg_mlops.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

