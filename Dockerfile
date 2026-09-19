# Multi-stage build for the credit risk PD service.

# ---------- Stage 1: build dependencies ----------
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        fastapi \
        uvicorn[standard] \
        pydantic \
        pandas \
        numpy \
        pyarrow \
        scikit-learn \
        lightgbm \
        shap \
        pyyaml

# ---------- Stage 2: runtime ----------
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy source
COPY src/ ./src/
COPY configs/ ./configs/
COPY artifacts/models/challenger/0.1.0/model.txt ./artifacts/models/challenger/0.1.0/model.txt
COPY artifacts/calibrators/challenger/calibrator.pkl ./artifacts/calibrators/challenger/calibrator.pkl

# Copy a small sample for the feature contract
COPY data/processed/challenger_holdout.parquet ./data/processed/challenger_holdout.parquet

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "credit_risk.api.main:app", "--host", "0.0.0.0", "--port", "8000"]