FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONUTF8=1 \
    PYTHONPATH=/app/src \
    FACTORY_DATA_DIR=/app/data \
    FACTORY_MODELS_DIR=/models \
    FACTORY_SEED_REGISTRY_DIR=/opt/factory-registry \
    FACTORY_SEED_DOCS_DIR=/opt/factory-docs \
    PATH=/app/.venv/bin:$PATH

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 git && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --no-cache-dir uv
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --inexact
COPY src ./src
COPY data/registry /opt/factory-registry
COPY data/docs /opt/factory-docs
EXPOSE 8000
CMD ["python", "-m", "factory", "serve", "--host", "0.0.0.0", "--port", "8000"]
