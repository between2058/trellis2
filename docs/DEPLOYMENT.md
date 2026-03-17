# TRELLIS.2 部署指南

## 系統需求

| 項目 | 最低需求 |
|------|----------|
| GPU | NVIDIA GPU, 24GB+ VRAM (A100/H100/RTX 4090) |
| CUDA | 12.4+ |
| Docker | 20.10+ |
| nvidia-container-toolkit | 已安裝並設定 |
| 磁碟空間 | ~30GB（Docker image）+ ~20GB（模型快取） |

---

## 快速部署

### 1. 設定環境變數

```bash
cp .env.example .env
```

編輯 `.env`：

```bash
# HuggingFace 模型快取路徑（主機上的絕對路徑）
HF_CACHE_PATH=/data/huggingface/cache

# GPU 裝置編號（0-based）
GPU_ID=0

# API 對外 port
API_PORT=52070

# （選填）HuggingFace token，用於 gated models
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

### 2. 建置並啟動

```bash
docker compose up -d --build
```

首次建置約需 30-60 分鐘（編譯 CUDA extensions）。後續啟動秒級。

### 3. 驗證

```bash
# 等待服務啟動（首次啟動會下載模型，約 5-10 分鐘）
curl http://localhost:52070/health

# 預期回傳：
# {"status":"ok","model_loaded":false,"gpu_busy":false}
# 模型會在首次推理請求時載入
```

---

## Docker 建置細節

### Dockerfile 結構

```
nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04
├── System packages (build-essential, cmake, ninja, ffmpeg, ...)
├── Python 3.10
├── PyTorch 2.7.1+cu128
├── requirements-api.txt (fastapi, trimesh, open3d, ...)
├── xformers 0.0.31
├── spconv-cu120
├── PyTorch re-lock（防止被其他套件降版）
├── nvdiffrast (from GitHub)
├── flash-attn 2.8.0.post2
└── Application source (trellis2/, trellis2_api.py)
```

### 建置參數

```bash
# 指定 GPU 架構（預設 sm_120 for Blackwell）
docker build --build-arg TORCH_CUDA_ARCH_LIST="8.9" -t trellis2-api .

# 常見架構：
# A100:     8.0
# H100:     9.0
# RTX 4090: 8.9
# RTX 5090: 12.0
# Blackwell Pro: 12.0
```

### 使用 Proxy

```bash
docker build \
  --build-arg http_proxy=http://proxy:8080 \
  --build-arg https_proxy=http://proxy:8080 \
  -t trellis2-api .
```

---

## docker-compose.yml 說明

```yaml
services:
  trellis2:
    ports:
      - "${API_PORT:-52070}:52070"     # API port mapping

    volumes:
      - ${HF_CACHE_PATH}:/app/hf_cache:rw  # 模型快取（避免重複下載）
      - ./logs:/app/logs                     # 日誌持久化

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${GPU_ID}"]      # 指定 GPU
              capabilities: [gpu]

    shm_size: "8gb"                          # PyTorch DataLoader 需要
    restart: unless-stopped                  # 自動重啟
```

---

## 多 GPU / 多實例部署

每個 GPU 跑一個獨立容器：

```bash
# GPU 0 on port 52070
GPU_ID=0 API_PORT=52070 docker compose -p trellis2-gpu0 up -d

# GPU 1 on port 52071
GPU_ID=1 API_PORT=52071 docker compose -p trellis2-gpu1 up -d
```

搭配 Nginx 做 load balancing：

```nginx
upstream trellis2 {
    server 127.0.0.1:52070;
    server 127.0.0.1:52071;
}

server {
    listen 80;
    location / {
        proxy_pass http://trellis2;
        client_max_body_size 50M;
    }
}
```

---

## 日誌管理

日誌位於 `./logs/`（透過 volume mount 持久化）：

| 檔案 | 內容 |
|------|------|
| `app.log` | 應用程式日誌（推理請求、GPU 記憶體、錯誤） |
| `access.log` | HTTP 存取日誌 |
| `uvicorn.log` | Uvicorn 伺服器日誌 |

- 自動每日輪替（midnight rotation）
- 保留 14 天
- 時區：UTC+8（台灣時間）
- `/health` 端點的存取記錄會自動過濾

---

## 監控

### Health Check

Docker 內建 health check：

```bash
docker inspect --format='{{.State.Health.Status}}' trellis2-api
```

- `interval`: 30s
- `timeout`: 15s
- `start-period`: 300s（等待模型載入）
- `retries`: 5

### GPU 記憶體

每次推理後會自動記錄 GPU 記憶體使用量到 `app.log`：

```
GPU [after generate]: allocated=12.34GB reserved=14.56GB
```

---

## 常見問題

### Q: 首次請求很慢？
**A:** 正常。模型會在首次推理時 lazy load（~20GB），載入約需 1-3 分鐘。後續請求不受影響。

### Q: GPU OOM？
**A:** 降低 `pipeline_type`（用 `512` 取代 `1024_cascade`）或降低 `texture_size`。1024_cascade 約需 20GB VRAM。

### Q: spconv 錯誤？
**A:** 確認 `SPCONV_ALGO=native` 環境變數已設定（docker-compose.yml 中已包含）。

### Q: 模型下載失敗？
**A:** 確認 `HF_CACHE_PATH` 指向有寫入權限的目錄。如需認證，設定 `HF_TOKEN` 環境變數。

### Q: 如何更新模型？
**A:** 刪除 `HF_CACHE_PATH` 中的快取目錄，重啟容器即會重新下載。

---

## 不使用 Docker 部署

如果不想使用 Docker，可以直接在主機上執行：

```bash
# 安裝依賴（假設已有 conda 環境）
pip install -r requirements-api.txt

# 啟動服務
SPCONV_ALGO=native python -m uvicorn trellis2_api:app \
  --host 0.0.0.0 \
  --port 52070 \
  --workers 1
```

注意：需要先完成 TRELLIS.2 的完整安裝（見 README.md 安裝步驟）。
