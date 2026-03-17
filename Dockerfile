# =============================================================================
# TRELLIS.2 API — Docker Image
#
# Follows the exact installation steps from README / setup.sh:
#   CUDA 12.4  |  Python 3.10  |  PyTorch 2.6.0+cu124
#
# Build:  docker build -t trellis2-api:latest .
# Run:    docker run --gpus all -p 52070:52070 trellis2-api:latest
# =============================================================================

FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ARG TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
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

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
    NVIDIA_VISIBLE_DEVICES=all \
    DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CUDA_HOME=/usr/local/cuda-12.4 \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    MAX_JOBS=${MAX_JOBS} \
    SPCONV_ALGO=native \
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache \
    HUGGINGFACE_HUB_CACHE=/app/hf_cache

# ── System packages ──────────────────────────────────────────────────────────
# setup.sh --basic requires libjpeg-dev for pillow-simd
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip \
    build-essential ninja-build cmake git wget curl \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev \
    libegl1-mesa-dev libgles2-mesa-dev libgomp1 ffmpeg libjpeg-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
 && python -m pip install --upgrade --no-cache-dir pip setuptools wheel

RUN test -d /usr/local/cuda-12.4 || ln -sf /usr/local/cuda /usr/local/cuda-12.4

WORKDIR /app

# ── setup.sh --new-env: PyTorch 2.6.0 + cu124 ───────────────────────────────
RUN pip install --no-cache-dir \
    torch==2.6.0 torchvision==0.21.0 \
    --index-url https://download.pytorch.org/whl/cu124

# ── setup.sh --basic ─────────────────────────────────────────────────────────
RUN pip install --no-cache-dir \
    imageio imageio-ffmpeg tqdm easydict opencv-python-headless ninja \
    trimesh transformers tensorboard pandas lpips zstandard \
    pillow-simd kornia timm \
    git+https://github.com/EasternJournalist/utils3d.git@9a4eb15e4021b67b12c460c7057d642626897ec8

# API deps (not in setup.sh, needed for FastAPI server)
COPY requirements-api.txt .
RUN pip install --no-cache-dir \
    fastapi==0.115.5 uvicorn[standard]==0.32.1 python-multipart==0.0.17 \
    pydantic>=2.0.0 open3d>=0.18.0

# ── setup.sh --flash-attn ────────────────────────────────────────────────────
RUN pip install --no-cache-dir flash-attn==2.7.3

# ── setup.sh --nvdiffrast ────────────────────────────────────────────────────
RUN mkdir -p /tmp/extensions \
 && git clone -b v0.4.0 https://github.com/NVlabs/nvdiffrast.git /tmp/extensions/nvdiffrast \
 && pip install /tmp/extensions/nvdiffrast --no-build-isolation

# ── setup.sh --nvdiffrec ─────────────────────────────────────────────────────
RUN git clone -b renderutils https://github.com/JeffreyXiang/nvdiffrec.git /tmp/extensions/nvdiffrec \
 && pip install /tmp/extensions/nvdiffrec --no-build-isolation

# ── setup.sh --cumesh ────────────────────────────────────────────────────────
RUN git clone https://github.com/JeffreyXiang/CuMesh.git /tmp/extensions/CuMesh --recursive \
 && pip install /tmp/extensions/CuMesh --no-build-isolation

# ── setup.sh --flexgemm ──────────────────────────────────────────────────────
RUN git clone https://github.com/JeffreyXiang/FlexGEMM.git /tmp/extensions/FlexGEMM --recursive \
 && pip install /tmp/extensions/FlexGEMM --no-build-isolation

# ── setup.sh --o-voxel ───────────────────────────────────────────────────────
# o-voxel is in-repo; its eigen submodule may not be checked out on server.
# Copy source then fetch eigen if the submodule dir is empty.
COPY o-voxel/ /tmp/extensions/o-voxel/
RUN if [ ! -f /tmp/extensions/o-voxel/third_party/eigen/Eigen/Dense ]; then \
        rm -rf /tmp/extensions/o-voxel/third_party/eigen && \
        git clone --depth 1 https://gitlab.com/libeigen/eigen.git \
            /tmp/extensions/o-voxel/third_party/eigen ; \
    fi \
 && pip install /tmp/extensions/o-voxel --no-build-isolation

# ── spconv ───────────────────────────────────────────────────────────────────
RUN pip install --no-cache-dir spconv-cu120

# ── xformers (optional, fallback attention backend) ──────────────────────────
RUN pip install --no-cache-dir xformers==0.0.29.post3

# ── Cleanup build artifacts ──────────────────────────────────────────────────
RUN rm -rf /tmp/extensions

# ── Application source ───────────────────────────────────────────────────────
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
