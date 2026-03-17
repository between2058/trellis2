# TRELLIS.2 Advanced Features & FastAPI Service Design

## Goal

Extend the TRELLIS.2 repo with all major features from ComfyUI-Trellis2 (multi-view generation, mesh processing, advanced samplers, staged pipelines), then serve it as a FastAPI service packaged in Docker following ReconViaGen conventions.

## Architecture Overview

```
trellis2/
├── pipelines/
│   ├── samplers/
│   │   ├── flow_euler.py              (existing)
│   │   ├── classifier_free_guidance_mixin.py (existing)
│   │   ├── guidance_interval_mixin.py  (existing)
│   │   ├── flow_euler_multiview.py     (NEW - multi-view spatial blending sampler)
│   │   └── flow_rk.py                 (NEW - RK4/Heun higher-order solvers)
│   ├── trellis2_image_to_3d.py        (EXTEND - add multiview methods)
│   ├── trellis2_texturing.py          (EXTEND - add multiview texturing)
│   └── base.py                        (existing, unchanged)
├── utils/
│   └── mesh_processing.py             (NEW - remesh, simplify, fill holes, smooth)
├── representations/
│   └── mesh/base.py                   (existing, unchanged)
trellis2_api.py                        (NEW - FastAPI server)
Dockerfile                             (NEW)
docker-compose.yml                     (NEW)
.env.example                           (NEW)
requirements-api.txt                   (NEW)
```

---

## Module 1: Multi-View Sampler

**File:** `trellis2/pipelines/samplers/flow_euler_multiview.py`

### Core Mechanism: Spatial View Weighting

Each voxel in a SparseTensor gets a weight per view based on its 3D position:

```
view_vectors = {
    'front': +z (or +x depending on front_axis),
    'back':  -z,
    'right': +x,
    'left':  -x,
}
weights = softmax(scores * blend_temperature, dim=views)
```

### Classes

1. `FlowEulerMultiViewSampler(FlowEulerSampler)` — base multi-view Euler sampler
2. `FlowEulerMultiViewCfgSampler` — with CFG support
3. `FlowEulerMultiViewGuidanceIntervalSampler` — with guidance interval

### Key Method: `sample_once_multiview()`

```python
def sample_once_multiview(self, model, x_t, t, t_prev, conds_dict, views, front_axis, blend_temperature, **kwargs):
    weights = compute_view_weights(x_t, views, front_axis, blend_temperature)
    pred_accum = 0
    for i, view in enumerate(views):
        pred = self._inference_model(model, x_t, t, cond=conds_dict[view], **kwargs)
        pred_accum += pred * weights[:, i:i+1]
    # Continue with standard flow step using blended prediction
```

### Supports Both Sparse and Dense Tensors

- **Sparse (SparseTensor):** weights from `coords[:, 1:]` normalized to [-1, 1]
- **Dense (torch.Tensor):** weights from meshgrid over full 3D volume

---

## Module 2: Multi-View Pipeline Methods

### In `trellis2_image_to_3d.py`:

```python
def run_multiview(
    self,
    front_image: Image.Image,
    back_image: Optional[Image.Image] = None,
    left_image: Optional[Image.Image] = None,
    right_image: Optional[Image.Image] = None,
    front_axis: str = 'z',
    blend_temperature: float = 2.0,
    # ... same params as run()
) -> List[MeshWithVoxel]:
```

Internally calls:
- `sample_sparse_structure_multiview(conds_dict, views, front_axis, blend_temperature, ...)`
- `sample_shape_slat_multiview(conds_dict, views, ...)`
- `sample_tex_slat_multiview(conds_dict, views, ...)`

Each `*_multiview` method:
1. Builds conds_dict from available views
2. Creates multi-view sampler instance
3. Calls `sample_multiview()` instead of `sample()`
4. Returns same types as single-view counterparts

### In `trellis2_texturing.py`:

```python
def run_multiview(
    self,
    mesh: trimesh.Trimesh,
    front_image: Image.Image,
    back_image: Optional[Image.Image] = None,
    left_image: Optional[Image.Image] = None,
    right_image: Optional[Image.Image] = None,
    front_axis: str = 'z',
    blend_temperature: float = 2.0,
    # ... same params as run()
) -> trimesh.Trimesh:
```

---

## Module 3: Mesh Processing Utilities

**File:** `trellis2/utils/mesh_processing.py`

All functions operate on `(vertices: Tensor, faces: Tensor)` or `trimesh.Trimesh`:

| Function | Backend | Purpose |
|----------|---------|---------|
| `remesh_dual_contouring(mesh, resolution, band, project_back)` | CuMesh | Rebuild topology via DC |
| `remesh_quad(mesh, resolution, band, project_back)` | CuMesh | Quad-dominant DC remesh |
| `reconstruct_mesh(mesh, resolution)` | CuMesh | Full volumetric reconstruction |
| `simplify_mesh(mesh, target_faces, method='cumesh')` | CuMesh/Meshlib | Reduce face count |
| `simplify_mesh_advanced(mesh, target_faces, **qem_params)` | CuMesh | QEM with tunable params |
| `fill_holes_meshlib(mesh)` | Meshlib (mrmeshpy) | Optimal hole triangulation |
| `fill_holes_cumesh(mesh, max_perimeter)` | CuMesh | Boundary-based hole fill |
| `smooth_normals(mesh)` | Trimesh | Smooth vertex normals |
| `laplacian_smooth(mesh, iterations, method)` | Open3D | Laplacian/Taubin smooth |
| `weld_vertices(mesh)` | Trimesh | Merge duplicate vertices |
| `remove_floaters(mesh)` | PyMeshLab | Remove disconnected components |

---

## Module 4: Advanced Samplers

**File:** `trellis2/pipelines/samplers/flow_rk.py`

### FlowRK4Sampler

4th-order Runge-Kutta ODE solver:
```
k1 = f(t, x)
k2 = f(t + h/2, x + h*k1/2)
k3 = f(t + h/2, x + h*k2/2)
k4 = f(t + h, x + h*k3)
x_next = x + h/6 * (k1 + 2*k2 + 2*k3 + k4)
```

### FlowHeunSampler

2nd-order predictor-corrector:
```
x_pred = x + h * f(t, x)
x_next = x + h/2 * (f(t, x) + f(t+h, x_pred))
```

Each has CFG, GuidanceInterval, and MultiView variants via mixin composition.

---

## Module 5: FastAPI Server

**File:** `trellis2_api.py`

### Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Health check (model_loaded, gpu_busy) |
| `/generate` | POST | Single image → 3D mesh (GLB) |
| `/generate-multiview` | POST | Multi-view images → 3D mesh (GLB) |
| `/texture` | POST | Mesh + image(s) → textured mesh (GLB) |
| `/texture-multiview` | POST | Mesh + multi-view images → textured mesh |
| `/mesh-process` | POST | Mesh processing (remesh, simplify, etc.) |
| `/download/{request_id}/{filename}` | GET | Download generated assets |

### Request/Response Patterns

```python
# /generate
class GenerateRequest(BaseModel):
    seed: int = 42
    pipeline_type: str = "1024_cascade"
    ss_guidance_strength: float = 7.5
    ss_sampling_steps: int = 30
    slat_guidance_strength: float = 3.0
    slat_sampling_steps: int = 12
    texture_size: int = 1024
    simplify: float = 0.95
    mesh_processing: Optional[MeshProcessingConfig] = None

# /generate-multiview
class GenerateMultiviewRequest(GenerateRequest):
    front_axis: str = "z"
    blend_temperature: float = 2.0

# /texture
class TextureRequest(BaseModel):
    seed: int = 42
    resolution: int = 1024
    texture_size: int = 2048
    guidance_strength: float = 3.0
    sampling_steps: int = 12

# /mesh-process
class MeshProcessRequest(BaseModel):
    operations: List[MeshOperation]  # ordered list of operations

class MeshOperation(BaseModel):
    type: str  # "remesh", "simplify", "fill_holes", "smooth_normals", etc.
    params: dict = {}
```

### Concurrency

- `asyncio.Lock()` for GPU access (single inference at a time)
- `run_in_threadpool()` for blocking inference calls
- Lazy model loading on first request

### File Handling

- Images: uploaded as multipart/form-data
- Meshes: uploaded as .glb/.obj files
- Outputs: stored in temp directory, served via `/download/`
- Cleanup: temp files removed after configurable TTL

---

## Module 6: Docker Packaging

### Dockerfile

Following ReconViaGen pattern:
- Base: `nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04`
- Multi-step build: system → PyTorch → CUDA extensions → app deps → source
- Re-lock PyTorch after all pip installs
- HEALTHCHECK on `/health` endpoint

### docker-compose.yml

```yaml
services:
  trellis2-api:
    build: .
    ports:
      - "${API_PORT:-52070}:52070"
    volumes:
      - ${HF_CACHE_PATH}:/app/hf_cache:rw
    environment:
      - SPCONV_ALGO=native
      - HF_HOME=/app/hf_cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${GPU_ID:-0}"]
              capabilities: [gpu]
    shm_size: "8gb"
    restart: unless-stopped
```

### .env.example

```
GPU_ID=0
API_PORT=52070
HF_CACHE_PATH=/path/to/huggingface/cache
HF_TOKEN=hf_xxx
TORCH_CUDA_ARCH_LIST=12.0
```

---

## Implementation Order

1. Multi-View Sampler (`flow_euler_multiview.py`)
2. Multi-View Pipeline Methods (extend existing files)
3. Mesh Processing Utilities (`mesh_processing.py`)
4. Advanced Samplers (`flow_rk.py`)
5. FastAPI Server (`trellis2_api.py`)
6. Docker Packaging (`Dockerfile`, `docker-compose.yml`, `.env.example`)

## Dependencies to Add

- fastapi, uvicorn, python-multipart (API)
- open3d (smoothing)
- pymeshlab (floater removal)
- mrmeshpy (hole filling via Meshlib)
- (cumesh, nvdiffrast, spconv already required by TRELLIS.2)
