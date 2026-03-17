# =============================================================================
# TRELLIS.2 API — Docker Image
#
# Target: NVIDIA RTX Pro 6000 (Blackwell, sm_120)
# CUDA: 12.8.1  |  Python: 3.10  |  PyTorch: 2.7.1+cu128
#
# Build:  docker build -t trellis2-api:latest .
# Run:    docker run --gpus all -p 52070:52070 trellis2-api:latest
# =============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="12.0"
ARG MAX_JOBS=4
ARG http_proxy=""
ARG https_proxy=""
ARG no_proxy="localhost,127.0.0.1"

ENV http_proxy=${http_proxy} \
    https_proxy=${https_proxy} \
    HTTP_PROXY=${http_proxy} \
    HTTPS_PROXY=${https_proxy} \
    no_proxy=${no_proxy} \
    NO_PROXY=${no_proxy}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-12.8 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    SPCONV_ALGO=native \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    HUGGINGFACE_HUB_CACHE=/app/hf_cache

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    build-essential ninja-build cmake git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libegl1-mesa-dev libgles2-mesa-dev libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

RUN test -d /usr/local/cuda-12.8 || ln -sf /usr/local/cuda /usr/local/cuda-12.8

WORKDIR /app

# Step 1: PyTorch
RUN pip install --no-cache-dir \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Step 2: Python deps
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Step 3: xformers
RUN pip install --no-cache-dir xformers==0.0.31

# Step 4: spconv
RUN pip install --no-cache-dir spconv-cu120

# Step 5: Re-lock PyTorch (prevent downgrades)
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# Step 6: nvdiffrast
RUN pip install --no-cache-dir --no-build-isolation \
    git+https://github.com/NVlabs/nvdiffrast.git

# Step 7: flash-attn
RUN MAX_JOBS=4 pip install --no-cache-dir --no-build-isolation \
    flash-attn==2.8.0.post2

# Application source
COPY trellis2/          /app/trellis2/
COPY trellis2_api.py    /app/trellis2_api.py

RUN mkdir -p /app/logs /app/outputs

EXPOSE 52070

HEALTHCHECK \
    --interval=30s \
    --timeout=15s \
    --start-period=300s \
    --retries=5 \
    CMD curl -f http://localhost:52070/health || exit 1

CMD ["python", "-m", "uvicorn", "trellis2_api:app", \
     "--host", "0.0.0.0", \
     "--port", "52070", \
     "--workers", "1", \
     "--log-level", "info"]
