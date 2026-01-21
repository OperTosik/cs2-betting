FROM python:3.11-slim

RUN apt-get update && \
    apt-get install -y build-essential git wget && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src
COPY datasets/ ./datasets
COPY models/ ./models
COPY config/ ./config

ENV PYTHONUNBUFFERED=1
ENV DATA_DIR=/app/data
ENV MODEL_DIR=/app/models
ENV PYTHONPATH=/app

# IF YOU HAVE PROBLEM WITH oneDNN
ENV USE_ONEDNN=0
