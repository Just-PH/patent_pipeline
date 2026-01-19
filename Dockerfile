FROM python:3.12-slim

# System deps minimales (tesseract + poppler pour pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    tesseract-ocr \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Poetry
RUN pip install --no-cache-dir poetry==2.1.1
RUN poetry config virtualenvs.create false

# Install deps (runtime only)
COPY pyproject.toml poetry.lock* /app/
RUN poetry install --only main --no-interaction --no-ansi --no-root

# Copy source
COPY src /app/src
COPY scripts /app/scripts
COPY README.md /app/README.md

# Ensure module import
ENV PYTHONPATH=/app/src

# HF cache dir (mount it in docker run)
ENV HF_HOME=/hf
ENV TRANSFORMERS_CACHE=/hf
ENV HF_HUB_DISABLE_TELEMETRY=1

ENTRYPOINT ["python"]
