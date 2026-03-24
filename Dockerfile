FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG BACKEND=doctr

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/root/.cache/huggingface

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential pkg-config \
    libtesseract-dev libleptonica-dev \
    libgomp1 poppler-utils libgl1 libglib2.0-0 \
    python3 python3-pip \
    tesseract-ocr tesseract-ocr-eng tesseract-ocr-deu tesseract-ocr-frk \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# pip tooling
RUN python -m pip install --no-cache-dir -U pip setuptools wheel

# NumPy <2 (important for paddle/opencv)
RUN python -m pip install --no-cache-dir "numpy<2.0"

# Torch CUDA (kept out of Poetry on purpose)
# Pin versions to avoid torchvision/torch mismatches.
RUN python -m pip install --no-cache-dir \
    "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "torchaudio==2.5.1+cu121" \
    --index-url https://download.pytorch.org/whl/cu121

# Poetry + deps
RUN python -m pip install --no-cache-dir poetry
COPY pyproject.toml poetry.lock /app/
RUN poetry config virtualenvs.create false && \
    if [ "$BACKEND" = "doctr" ]; then \
        poetry install --only main,bench,surya,doctr --no-interaction --no-ansi --no-root ; \
        python -m pip install --no-cache-dir "tesserocr>=2.8,<3.0" ; \
    elif [ "$BACKEND" = "lightonocr" ]; then \
        poetry install --only main,bench,lightonocr --no-interaction --no-ansi --no-root ; \
        # Install transformers from git (LightOnOCR support)
        python -m pip install --no-cache-dir "git+https://github.com/huggingface/transformers.git" ; \
        # Re-align torch/torchvision/torchaudio after poetry+transformers installs.
        # This avoids ABI mismatches like missing torchvision::nms.
        python -m pip install --no-cache-dir --force-reinstall \
          "torch==2.5.1+cu121" "torchvision==0.20.1+cu121" "torchaudio==2.5.1+cu121" \
          --index-url https://download.pytorch.org/whl/cu121 ; \
        # Build-time sanity check for LightOnOCR
        python -c "import torch, torchvision, transformers; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('transformers:', transformers.__version__); from transformers import LightOnOcrForConditionalGeneration, LightOnOcrProcessor; print('LightOnOCR OK')" ; \
    elif [ "$BACKEND" = "slm" ]; then \
        # SLM-only image (no OCR backends beyond core deps)
        poetry install --only main,bench --no-interaction --no-ansi --no-root ; \
        # Re-pin torch stack after poetry, then remove torchvision (not needed for text-only SLM).
        # This avoids torchvision/torch ABI issues such as missing torchvision::nms.
        python -m pip install --no-cache-dir --force-reinstall \
          "torch==2.5.1+cu121" "torchaudio==2.5.1+cu121" \
          --index-url https://download.pytorch.org/whl/cu121 ; \
        python -m pip uninstall -y torchvision || true ; \
        python -c "import importlib.util, transformers, torch; print('transformers:', transformers.__version__); print('torch:', torch.__version__); print('torchvision_installed:', importlib.util.find_spec('torchvision') is not None); from transformers import Mistral3ForConditionalGeneration; print('Mistral3 import OK')" ; \
    else \
        echo "Unknown BACKEND=$BACKEND (expected doctr|lightonocr|slm)" && exit 1 ; \
    fi

# Final ABI lock for OpenCV/NumPy compatibility (prevents NumPy 2.x + cv2 ABI mismatch).
RUN python -m pip install --no-cache-dir --force-reinstall \
    "numpy<2" \
    "opencv-python-headless==4.10.0.84" && \
    python -c "import numpy as np, cv2; print('numpy:', np.__version__); print('cv2:', cv2.__version__)"

# Code
COPY src /app/src
COPY scripts /app/scripts

CMD ["python"]
