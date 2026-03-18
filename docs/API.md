# TRELLIS.2 API Reference

TRELLIS.2 FastAPI 服務提供 RESTful 端點，用於 3D 模型生成、貼圖與 Mesh 後處理。

**Base URL:** `http://<host>:52070`

---

## Endpoints

### `GET /health`

健康檢查端點。

**Response:**

```json
{
  "status": "ok",
  "model_loaded": true,
  "gpu_busy": false
}
```

| 欄位 | 型別 | 說明 |
|------|------|------|
| `status` | string | 服務狀態 |
| `model_loaded` | bool | 模型是否已載入 |
| `gpu_busy` | bool | GPU 是否正在執行推理 |

---

### `POST /generate`

單張圖片生成 3D mesh (GLB)。

**Request:** `multipart/form-data`

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `file` | file | (必填) | 輸入圖片 (PNG/JPG) |
| `seed` | int | 42 | 隨機種子 |
| `pipeline_type` | string | `"1024_cascade"` | 管線類型：`512`, `1024`, `1024_cascade`, `1536_cascade` |
| `ss_guidance_strength` | float | 7.5 | Sparse Structure 引導強度 |
| `ss_sampling_steps` | int | 30 | Sparse Structure 取樣步數 |
| `slat_guidance_strength` | float | 3.0 | Shape/Texture latent 引導強度 |
| `slat_sampling_steps` | int | 12 | Shape/Texture latent 取樣步數 |
| `texture_size` | int | 1024 | 輸出貼圖解析度 |

**Response:**

```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "glb_url": "/download/550e8400-e29b-41d4-a716-446655440000/output.glb"
}
```

**cURL 範例:**

```bash
curl -X POST http://localhost:52070/generate \
  -F "file=@input.png" \
  -F "seed=42" \
  -F "pipeline_type=1024_cascade"
```

---

### `POST /generate-multiview`

多視角圖片生成 3D mesh，透過空間混合（spatial blending）提升品質。

**Request:** `multipart/form-data`

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `front` | file | (必填) | 正面圖片 |
| `back` | file | null | 背面圖片 |
| `left` | file | null | 左側圖片 |
| `right` | file | null | 右側圖片 |
| `seed` | int | 42 | 隨機種子 |
| `pipeline_type` | string | `"1024_cascade"` | 管線類型 |
| `ss_guidance_strength` | float | 7.5 | Sparse Structure 引導強度 |
| `ss_sampling_steps` | int | 30 | Sparse Structure 取樣步數 |
| `slat_guidance_strength` | float | 3.0 | Latent 引導強度 |
| `slat_sampling_steps` | int | 12 | Latent 取樣步數 |
| `front_axis` | string | `"z"` | 正面軸向：`z` 或 `x` |
| `blend_temperature` | float | 2.0 | 混合溫度（越高越平滑，越低越銳利） |
| `sampler` | string | `"euler"` | ODE 求解器：`euler`, `rk4`, `heun` |

**Response:** 同 `/generate`

**cURL 範例:**

```bash
curl -X POST http://localhost:52070/generate-multiview \
  -F "front=@front.png" \
  -F "back=@back.png" \
  -F "left=@left.png" \
  -F "right=@right.png" \
  -F "sampler=rk4" \
  -F "blend_temperature=2.0"
```

**Multi-View 空間混合機制：**

每個 voxel 根據其 3D 座標計算對各視角的相關性權重：
- 位於物體正面的 voxel → 正面視角權重較高
- 位於物體背面的 voxel → 背面視角權重較高
- 權重透過 softmax 正規化，`blend_temperature` 控制分佈銳利度

---

### `POST /texture`

對已有的 mesh 進行 PBR 貼圖生成。

**Request:** `multipart/form-data`

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `file` | file | (必填) | 參考圖片 |
| `mesh_file` | file | (必填) | Mesh 檔案 (.glb/.obj) |
| `seed` | int | 42 | 隨機種子 |
| `resolution` | int | 1024 | 處理解析度 |
| `texture_size` | int | 2048 | 輸出貼圖解析度 |

**Response:** 同 `/generate`

**cURL 範例:**

```bash
curl -X POST http://localhost:52070/texture \
  -F "file=@reference.png" \
  -F "mesh_file=@model.glb" \
  -F "texture_size=2048"
```

**Multi-Part GLB 支援：**

當輸入的 mesh 為多零件 GLB 時，API 自動偵測並：
1. 套用各零件的 scene graph transform
2. 合併後統一生成 PBR voxel（GPU 只跑一次）
3. 每個零件獨立 UV unwrap + 貼圖取樣
4. 還原各零件原始座標
5. 輸出保留零件結構的多零件 GLB

---

### `POST /texture-multiview`

多視角參考圖片對 mesh 進行貼圖。

**Request:** `multipart/form-data`

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `front` | file | (必填) | 正面參考圖 |
| `mesh_file` | file | (必填) | Mesh 檔案 (.glb/.obj) |
| `back` | file | null | 背面參考圖 |
| `left` | file | null | 左側參考圖 |
| `right` | file | null | 右側參考圖 |
| `seed` | int | 42 | 隨機種子 |
| `resolution` | int | 1024 | 處理解析度 |
| `texture_size` | int | 2048 | 輸出貼圖解析度 |
| `front_axis` | string | `"z"` | 正面軸向 |
| `blend_temperature` | float | 2.0 | 混合溫度 |

**Response:** 同 `/generate`

**Multi-Part GLB 支援：** 同 `/texture`，自動偵測多零件 GLB 並保留零件結構。

---

### Mesh 後處理端點

所有 mesh 後處理操作各自獨立為單一端點，每個端點只包含該操作所需的參數。

**共用回應格式：**

```json
{
  "request_id": "...",
  "glb_url": "/download/.../output.glb",
  "vertices": 50000,
  "faces": 100000
}
```

---

#### `POST /mesh-process/simplify`

減少 mesh 面數，保持形狀。

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `mesh_file` | file | (必填) | Mesh 檔案 (.glb/.obj) |
| `target_face_num` | int | 100000 | 目標面數 |
| `method` | string | `"cumesh"` | 後端：`cumesh` (GPU) 或 `meshlib` (CPU) |

```bash
curl -X POST http://localhost:52070/mesh-process/simplify \
  -F "mesh_file=@model.glb" \
  -F "target_face_num=50000"
```

---

#### `POST /mesh-process/remesh`

使用 Dual Contouring 重建 mesh 拓撲。

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `mesh_file` | file | (必填) | Mesh 檔案 |
| `resolution` | int | 512 | Grid 解析度 |

```bash
curl -X POST http://localhost:52070/mesh-process/remesh \
  -F "mesh_file=@model.glb" \
  -F "resolution=512"
```

---

#### `POST /mesh-process/fill-holes`

填補 mesh 孔洞。

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `mesh_file` | file | (必填) | Mesh 檔案 |
| `method` | string | `"meshlib"` | 後端：`meshlib`（最佳三角化）或 `cumesh`（邊界法） |

```bash
curl -X POST http://localhost:52070/mesh-process/fill-holes \
  -F "mesh_file=@model.glb" \
  -F "method=meshlib"
```

---

#### `POST /mesh-process/smooth-normals`

計算平滑頂點法線。無額外參數。

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `mesh_file` | file | (必填) | Mesh 檔案 |

```bash
curl -X POST http://localhost:52070/mesh-process/smooth-normals \
  -F "mesh_file=@model.glb"
```

---

#### `POST /mesh-process/laplacian-smooth`

Laplacian 或 Taubin 幾何平滑。

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `mesh_file` | file | (必填) | Mesh 檔案 |
| `iterations` | int | 5 | 平滑迭代次數 |
| `method` | string | `"laplacian"` | 平滑方法：`laplacian` 或 `taubin` |

```bash
curl -X POST http://localhost:52070/mesh-process/laplacian-smooth \
  -F "mesh_file=@model.glb" \
  -F "iterations=10" \
  -F "method=taubin"
```

---

#### `POST /mesh-process/weld-vertices`

合併重複頂點。無額外參數。

| 參數 | 型別 | 預設 | 說明 |
|------|------|------|------|
| `mesh_file` | file | (必填) | Mesh 檔案 |

```bash
curl -X POST http://localhost:52070/mesh-process/weld-vertices \
  -F "mesh_file=@model.glb"
```

---

### `GET /download/{request_id}/{file_name}`

下載生成的檔案。

**Path Parameters:**

| 參數 | 說明 |
|------|------|
| `request_id` | 請求 ID（由生成端點回傳） |
| `file_name` | 檔案名稱（通常為 `output.glb`） |

---

## Sampler 比較

| Sampler | 階數 | NFE/步 | 品質 | 速度 | 建議用途 |
|---------|------|--------|------|------|----------|
| `euler` | 1st | 1 | 好 | 最快 | 一般用途、快速預覽 |
| `heun` | 2nd | 2 | 更好 | 中等 | 品質與速度平衡 |
| `rk4` | 4th | 4 | 最佳 | 最慢 | 最高品質需求 |

---

## 錯誤處理

所有端點在發生錯誤時回傳：

```json
{
  "detail": "錯誤訊息描述"
}
```

HTTP 狀態碼：
- `400` — 請求參數錯誤
- `404` — 檔案不存在
- `500` — 伺服器內部錯誤（GPU OOM、模型錯誤等）

---

## 並行處理

- API 使用 `asyncio.Lock()` 確保同一時間只有一個 GPU 推理任務
- 多個請求會自動排隊等待
- 模型在首次請求時延遲載入（lazy loading）
