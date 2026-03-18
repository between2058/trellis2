import os
import shutil
import tempfile
import uuid
import gc
import asyncio
import datetime
import logging
import logging.handlers
import torch
import numpy as np
from PIL import Image
from typing import Optional

os.environ['SPCONV_ALGO'] = 'native'

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.pipelines.trellis2_texturing import Trellis2TexturingPipeline
import o_voxel
import trimesh as _trimesh


# =============================================================================
# Logging
# =============================================================================

os.makedirs("logs", exist_ok=True)


class TaiwanFormatter(logging.Formatter):
    _TZ = datetime.timezone(datetime.timedelta(hours=8))
    def formatTime(self, record, datefmt=None):
        dt = datetime.datetime.fromtimestamp(record.created, tz=self._TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.strftime("%Y-%m-%d %H:%M:%S") + f",{record.msecs:03.0f}"


class HealthCheckFilter(logging.Filter):
    def filter(self, record):
        return "GET /health" not in record.getMessage()


def _rotating_handler(filename, formatter):
    h = logging.handlers.TimedRotatingFileHandler(
        f"logs/{filename}", when="midnight", backupCount=14, encoding="utf-8",
    )
    h.setFormatter(formatter)
    return h


_fmt = TaiwanFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
_access_fmt = TaiwanFormatter("%(asctime)s %(message)s")

logger = logging.getLogger("trellis2_api")
logger.setLevel(logging.DEBUG)
logger.propagate = False
logger.addHandler(_rotating_handler("app.log", _fmt))
logger.addHandler(logging.StreamHandler())

logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())
logging.getLogger("uvicorn.access").addHandler(_rotating_handler("access.log", _access_fmt))
logging.getLogger("uvicorn").addHandler(_rotating_handler("uvicorn.log", _fmt))

# =============================================================================
# App
# =============================================================================

app = FastAPI(title="TRELLIS.2 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

OUTPUT_DIR = tempfile.mkdtemp()
logger.info(f"Output directory: {OUTPUT_DIR}")

pipeline = None
tex_pipeline = None
gpu_lock = asyncio.Lock()


def export_mesh_to_glb(mesh, glb_path, texture_size=4096, decimation_target=1000000):
    """Convert MeshWithVoxel to GLB via o_voxel.postprocess."""
    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices,
        faces=mesh.faces,
        attr_volume=mesh.attrs,
        coords=mesh.coords,
        attr_layout=mesh.layout,
        voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=decimation_target,
        texture_size=texture_size,
        remesh=True,
        remesh_band=1,
        remesh_project=0,
        verbose=True,
    )
    glb.export(glb_path, extension_webp=True)


def load_glb_parts(path: str):
    """Load GLB and return (parts, merged_mesh, is_multipart).

    Groups by scene graph *node* (not by primitive), so a mesh with
    multiple materials stays as one part.  Each part gets its node's
    world transform baked in.
    """
    loaded = _trimesh.load(path, process=False)
    if isinstance(loaded, _trimesh.Trimesh):
        return [loaded], loaded, False
    if isinstance(loaded, _trimesh.Scene):
        # Group primitives by their scene-graph node.
        # A single node can reference a geometry that was split into
        # multiple primitives by trimesh; we merge those back together.
        node_meshes = {}  # node_name -> list[Trimesh]
        for node_name in loaded.graph.nodes_geometry:
            transform, geom_name = loaded.graph[node_name]
            geom = loaded.geometry[geom_name]
            if not isinstance(geom, _trimesh.Trimesh):
                continue
            mesh_with_tf = geom.copy().apply_transform(transform)
            node_meshes.setdefault(node_name, []).append(mesh_with_tf)

        if not node_meshes:
            raise ValueError("No mesh geometry found in file")

        # Merge primitives that belong to the same node
        parts = []
        for node_name, meshes in node_meshes.items():
            if len(meshes) == 1:
                parts.append(meshes[0])
            else:
                parts.append(_trimesh.util.concatenate(meshes))

        if len(parts) <= 1:
            merged = parts[0] if parts else None
            return parts, merged, False

        for i, p in enumerate(parts):
            ctr = (p.vertices.max(axis=0) + p.vertices.min(axis=0)) / 2
            logger.info(f"  Part {i}: {len(p.vertices)} verts, center={ctr.tolist()}")
        merged = _trimesh.util.concatenate(parts)
        logger.info(f"Merged {len(parts)} parts -> {len(merged.vertices)} verts")
        return parts, merged, True
    raise ValueError(f"Unexpected type from trimesh.load: {type(loaded)}")


def split_textured_mesh(textured_mesh, part_face_counts):
    """Split a textured mesh back into parts using face count boundaries.

    UV unwrap preserves face count and order, so the face boundaries from
    the original concatenated parts can be used to split the textured output.

    Returns None if face count mismatch (UV unwrap changed topology).
    """
    total_expected = sum(part_face_counts)
    actual = len(textured_mesh.faces)
    if actual != total_expected:
        logger.warning(
            f"Face count mismatch after texturing: expected {total_expected}, got {actual}. "
            f"Skipping split — returning single textured mesh."
        )
        return None

    faces = textured_mesh.faces
    vertices = textured_mesh.vertices
    normals = textured_mesh.vertex_normals.copy()

    has_visual = (isinstance(textured_mesh.visual, _trimesh.visual.TextureVisuals)
                  and textured_mesh.visual.uv is not None)
    uvs = textured_mesh.visual.uv if has_visual else None
    material = textured_mesh.visual.material if has_visual else None

    parts = []
    face_offset = 0
    for i, count in enumerate(part_face_counts):
        part_faces = faces[face_offset:face_offset + count]
        if len(part_faces) == 0:
            logger.warning(f"Part {i} has 0 faces, skipping")
            face_offset += count
            continue
        unique_verts, inverse = np.unique(part_faces.ravel(), return_inverse=True)
        new_faces = inverse.reshape(-1, 3)
        new_vertices = vertices[unique_verts]

        kwargs = dict(vertices=new_vertices, faces=new_faces, process=False)
        if len(unique_verts) > 0 and len(normals) > 0:
            kwargs['vertex_normals'] = normals[unique_verts]
        if has_visual and len(unique_verts) > 0:
            kwargs['visual'] = _trimesh.visual.TextureVisuals(
                uv=uvs[unique_verts], material=material)

        parts.append(_trimesh.Trimesh(**kwargs))
        face_offset += count

    return parts


def texture_multipart(pipe, parts, merged_mesh, image, seed=42,
                      resolution=1024, texture_size=2048):
    """Texture a multi-part mesh: merge → texture once → split by face boundaries.

    1. Record per-part face counts
    2. Run full pipeline once on merged mesh
    3. Restore original coordinates (preprocess_mesh centers+scales, postprocess only reverses axis swap)
    4. Split textured mesh by face count boundaries
    5. Assemble into trimesh.Scene
    """
    part_face_counts = [len(p.faces) for p in parts]
    logger.info(f"Multipart texture: {len(parts)} parts, face counts={part_face_counts}")

    # Compute center/scale (same formula as pipeline.preprocess_mesh)
    verts = merged_mesh.vertices
    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()

    # Run full pipeline once on merged mesh
    textured = pipe.run(merged_mesh, image, seed=seed,
                        resolution=resolution, texture_size=texture_size)

    # Split by face boundaries (before coord restore to preserve normals cache)
    textured_parts = split_textured_mesh(textured, part_face_counts)

    if textured_parts is None:
        # Face count mismatch — return single textured mesh with restored coords
        textured.vertices = textured.vertices / scale + center
        return textured

    # Restore original coordinates on each part
    for part in textured_parts:
        part.vertices = part.vertices / scale + center

    scene = _trimesh.Scene()
    for i, part in enumerate(textured_parts):
        scene.add_geometry(part, node_name=f"part_{i}")
    return scene


def texture_multipart_multiview(pipe, parts, merged_mesh,
                                front_image, back_image=None,
                                left_image=None, right_image=None,
                                seed=42, resolution=1024, texture_size=2048,
                                front_axis='z', blend_temperature=2.0):
    """Multi-view version of texture_multipart: merge → texture once → split."""
    part_face_counts = [len(p.faces) for p in parts]
    logger.info(f"Multipart texture (multiview): {len(parts)} parts, face counts={part_face_counts}")

    verts = merged_mesh.vertices
    vmin, vmax = verts.min(axis=0), verts.max(axis=0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()

    textured = pipe.run_multiview(
        merged_mesh, front_image=front_image, back_image=back_image,
        left_image=left_image, right_image=right_image,
        seed=seed, resolution=resolution, texture_size=texture_size,
        front_axis=front_axis, blend_temperature=blend_temperature,
    )

    textured_parts = split_textured_mesh(textured, part_face_counts)

    if textured_parts is None:
        textured.vertices = textured.vertices / scale + center
        return textured

    for part in textured_parts:
        part.vertices = part.vertices / scale + center

    scene = _trimesh.Scene()
    for i, part in enumerate(textured_parts):
        scene.add_geometry(part, node_name=f"part_{i}")
    return scene


def log_gpu_memory(label: str):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU [{label}]: allocated={alloc:.2f}GB reserved={reserved:.2f}GB")


def ensure_model_loaded():
    global pipeline, tex_pipeline
    if pipeline is None:
        logger.info("Loading TRELLIS.2 models...")
        log_gpu_memory("before load")
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
        pipeline.cuda()
        log_gpu_memory("pipeline loaded")
        logger.info("TRELLIS.2 pipeline loaded.")

    if tex_pipeline is None:
        logger.info("Loading texturing pipeline...")
        tex_pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B", config_file="texturing_pipeline.json")
        tex_pipeline.cuda()
        log_gpu_memory("tex pipeline loaded")
        logger.info("Texturing pipeline loaded.")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "model_loaded": pipeline is not None,
        "gpu_busy": gpu_lock.locked(),
    }


@app.post("/generate")
async def generate(
    file: UploadFile = File(...),
    seed: int = Form(42),
    pipeline_type: str = Form("1024_cascade"),
    ss_guidance_strength: float = Form(7.5),
    ss_sampling_steps: int = Form(30),
    slat_guidance_strength: float = Form(3.0),
    slat_sampling_steps: int = Form(12),
    texture_size: int = Form(1024),
):
    """Single image to 3D mesh."""
    try:
        request_id = str(uuid.uuid4())
        req_dir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(req_dir, exist_ok=True)

        input_path = os.path.join(req_dir, "input.png")
        with open(input_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        image = Image.open(input_path)

        async with gpu_lock:
            await run_in_threadpool(ensure_model_loaded)
            logger.info(f"[{request_id}] Generating (single image)...")

            def _run():
                results = pipeline.run(
                    image,
                    seed=seed,
                    pipeline_type=pipeline_type,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "guidance_strength": ss_guidance_strength,
                    },
                    shape_slat_sampler_params={
                        "steps": slat_sampling_steps,
                        "guidance_strength": slat_guidance_strength,
                    },
                )
                mesh = results[0]
                glb_path = os.path.join(req_dir, "output.glb")
                export_mesh_to_glb(mesh, glb_path, texture_size=texture_size)
                return glb_path

            glb_path = await run_in_threadpool(_run)
            log_gpu_memory("after generate")
            torch.cuda.empty_cache()

        return {
            "request_id": request_id,
            "glb_url": f"/download/{request_id}/output.glb",
        }
    except Exception as e:
        logger.error(f"Generate failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-multiview")
async def generate_multiview(
    front: UploadFile = File(...),
    back: Optional[UploadFile] = File(None),
    left: Optional[UploadFile] = File(None),
    right: Optional[UploadFile] = File(None),
    seed: int = Form(42),
    pipeline_type: str = Form("1024_cascade"),
    ss_guidance_strength: float = Form(7.5),
    ss_sampling_steps: int = Form(30),
    slat_guidance_strength: float = Form(3.0),
    slat_sampling_steps: int = Form(12),
    front_axis: str = Form("z"),
    blend_temperature: float = Form(2.0),
    sampler: str = Form("euler"),
):
    """Multi-view images to 3D mesh."""
    try:
        request_id = str(uuid.uuid4())
        req_dir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(req_dir, exist_ok=True)

        def _save_upload(upload, name):
            if upload is None:
                return None
            path = os.path.join(req_dir, f"{name}.png")
            with open(path, "wb") as buf:
                shutil.copyfileobj(upload.file, buf)
            return Image.open(path)

        front_img = _save_upload(front, "front")
        back_img = _save_upload(back, "back")
        left_img = _save_upload(left, "left")
        right_img = _save_upload(right, "right")

        async with gpu_lock:
            await run_in_threadpool(ensure_model_loaded)
            logger.info(f"[{request_id}] Generating (multi-view)...")

            def _run():
                results = pipeline.run_multiview(
                    front_image=front_img,
                    back_image=back_img,
                    left_image=left_img,
                    right_image=right_img,
                    seed=seed,
                    pipeline_type=pipeline_type,
                    sparse_structure_sampler_params={
                        "steps": ss_sampling_steps,
                        "guidance_strength": ss_guidance_strength,
                    },
                    shape_slat_sampler_params={
                        "steps": slat_sampling_steps,
                        "guidance_strength": slat_guidance_strength,
                    },
                    front_axis=front_axis,
                    blend_temperature=blend_temperature,
                    sampler=sampler,
                )
                mesh = results[0]
                glb_path = os.path.join(req_dir, "output.glb")
                export_mesh_to_glb(mesh, glb_path)
                return glb_path

            glb_path = await run_in_threadpool(_run)
            log_gpu_memory("after generate-multiview")
            torch.cuda.empty_cache()

        return {
            "request_id": request_id,
            "glb_url": f"/download/{request_id}/output.glb",
        }
    except Exception as e:
        logger.error(f"Generate-multiview failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/texture")
async def texture_mesh(
    file: UploadFile = File(..., description="Reference image"),
    mesh_file: UploadFile = File(..., description="Mesh file (.glb/.obj)"),
    seed: int = Form(42),
    resolution: int = Form(1024),
    texture_size: int = Form(2048),
):
    """Texture an existing mesh with a reference image."""
    try:
        request_id = str(uuid.uuid4())
        req_dir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(req_dir, exist_ok=True)

        img_path = os.path.join(req_dir, "ref.png")
        with open(img_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)
        image = Image.open(img_path)

        mesh_path = os.path.join(req_dir, f"input_mesh{os.path.splitext(mesh_file.filename)[1]}")
        with open(mesh_path, "wb") as buf:
            shutil.copyfileobj(mesh_file.file, buf)
        parts, merged, is_multipart = load_glb_parts(mesh_path)

        async with gpu_lock:
            await run_in_threadpool(ensure_model_loaded)
            logger.info(f"[{request_id}] Texturing mesh (multipart={is_multipart}, {len(parts)} parts)...")

            def _run():
                out_path = os.path.join(req_dir, "output.glb")
                if is_multipart:
                    scene = texture_multipart(
                        tex_pipeline, parts, merged, image,
                        seed=seed, resolution=resolution, texture_size=texture_size,
                    )
                    scene.export(out_path)
                else:
                    result = tex_pipeline.run(
                        merged, image, seed=seed,
                        resolution=resolution, texture_size=texture_size,
                    )
                    result.export(out_path)
                return out_path

            out_path = await run_in_threadpool(_run)
            log_gpu_memory("after texture")
            torch.cuda.empty_cache()

        return {
            "request_id": request_id,
            "glb_url": f"/download/{request_id}/output.glb",
        }
    except Exception as e:
        logger.error(f"Texture failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/texture-multiview")
async def texture_mesh_multiview(
    front: UploadFile = File(...),
    mesh_file: UploadFile = File(...),
    back: Optional[UploadFile] = File(None),
    left: Optional[UploadFile] = File(None),
    right: Optional[UploadFile] = File(None),
    seed: int = Form(42),
    resolution: int = Form(1024),
    texture_size: int = Form(2048),
    front_axis: str = Form("z"),
    blend_temperature: float = Form(2.0),
):
    """Texture an existing mesh with multi-view reference images."""
    try:
        request_id = str(uuid.uuid4())
        req_dir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(req_dir, exist_ok=True)

        def _save(upload, name):
            if upload is None:
                return None
            path = os.path.join(req_dir, f"{name}.png")
            with open(path, "wb") as buf:
                shutil.copyfileobj(upload.file, buf)
            return Image.open(path)

        front_img = _save(front, "front")
        back_img = _save(back, "back")
        left_img = _save(left, "left")
        right_img = _save(right, "right")

        mesh_path = os.path.join(req_dir, f"input_mesh{os.path.splitext(mesh_file.filename)[1]}")
        with open(mesh_path, "wb") as buf:
            shutil.copyfileobj(mesh_file.file, buf)
        parts, merged, is_multipart = load_glb_parts(mesh_path)

        async with gpu_lock:
            await run_in_threadpool(ensure_model_loaded)
            logger.info(f"[{request_id}] Texturing mesh (multi-view, multipart={is_multipart}, {len(parts)} parts)...")

            def _run():
                out_path = os.path.join(req_dir, "output.glb")
                if is_multipart:
                    scene = texture_multipart_multiview(
                        tex_pipeline, parts, merged,
                        front_image=front_img, back_image=back_img,
                        left_image=left_img, right_image=right_img,
                        seed=seed, resolution=resolution, texture_size=texture_size,
                        front_axis=front_axis, blend_temperature=blend_temperature,
                    )
                    scene.export(out_path)
                else:
                    result = tex_pipeline.run_multiview(
                        merged, front_image=front_img, back_image=back_img,
                        left_image=left_img, right_image=right_img,
                        seed=seed, resolution=resolution, texture_size=texture_size,
                        front_axis=front_axis, blend_temperature=blend_temperature,
                    )
                    result.export(out_path)
                return out_path

            out_path = await run_in_threadpool(_run)
            log_gpu_memory("after texture-multiview")
            torch.cuda.empty_cache()

        return {
            "request_id": request_id,
            "glb_url": f"/download/{request_id}/output.glb",
        }
    except Exception as e:
        logger.error(f"Texture-multiview failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


async def _load_and_process_mesh(mesh_file: UploadFile, process_fn):
    """Shared helper: load mesh, apply process_fn, export result."""

    request_id = str(uuid.uuid4())
    req_dir = os.path.join(OUTPUT_DIR, request_id)
    os.makedirs(req_dir, exist_ok=True)

    mesh_path = os.path.join(req_dir, f"input{os.path.splitext(mesh_file.filename)[1]}")
    with open(mesh_path, "wb") as buf:
        shutil.copyfileobj(mesh_file.file, buf)
    mesh = _trimesh.load(mesh_path, force='mesh')

    result = await run_in_threadpool(process_fn, mesh)

    out_path = os.path.join(req_dir, "output.glb")
    result.export(out_path)

    return {
        "request_id": request_id,
        "glb_url": f"/download/{request_id}/output.glb",
        "vertices": len(result.vertices),
        "faces": len(result.faces),
    }


@app.post("/mesh-process/simplify")
async def mesh_simplify(
    mesh_file: UploadFile = File(...),
    target_face_num: int = Form(100000, description="Target number of faces after simplification"),
    method: str = Form("cumesh", description="Backend: 'cumesh' (GPU) or 'meshlib' (CPU)"),
):
    """Reduce mesh face count while preserving shape."""
    from trellis2.utils.mesh_processing import simplify_mesh
    try:
        return await _load_and_process_mesh(
            mesh_file, lambda m: simplify_mesh(m, target_face_num, method=method)
        )
    except Exception as e:
        logger.error(f"Simplify failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mesh-process/remesh")
async def mesh_remesh(
    mesh_file: UploadFile = File(...),
    resolution: int = Form(512, description="Dual contouring grid resolution"),
):
    """Rebuild mesh topology using dual contouring."""
    from trellis2.utils.mesh_processing import remesh_dual_contouring
    try:
        return await _load_and_process_mesh(
            mesh_file, lambda m: remesh_dual_contouring(m, resolution=resolution)
        )
    except Exception as e:
        logger.error(f"Remesh failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mesh-process/fill-holes")
async def mesh_fill_holes(
    mesh_file: UploadFile = File(...),
    method: str = Form("meshlib", description="Backend: 'meshlib' (optimal triangulation) or 'cumesh' (boundary-based)"),
):
    """Fill holes in mesh."""
    from trellis2.utils.mesh_processing import fill_holes
    try:
        return await _load_and_process_mesh(
            mesh_file, lambda m: fill_holes(m, method=method)
        )
    except Exception as e:
        logger.error(f"Fill holes failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mesh-process/smooth-normals")
async def mesh_smooth_normals(
    mesh_file: UploadFile = File(...),
):
    """Compute smooth vertex normals."""
    from trellis2.utils.mesh_processing import smooth_normals
    try:
        return await _load_and_process_mesh(mesh_file, smooth_normals)
    except Exception as e:
        logger.error(f"Smooth normals failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mesh-process/laplacian-smooth")
async def mesh_laplacian_smooth(
    mesh_file: UploadFile = File(...),
    iterations: int = Form(5, description="Number of smoothing iterations"),
    method: str = Form("laplacian", description="Smoothing method: 'laplacian' or 'taubin'"),
):
    """Smooth mesh geometry using Laplacian or Taubin smoothing."""
    from trellis2.utils.mesh_processing import laplacian_smooth
    try:
        return await _load_and_process_mesh(
            mesh_file, lambda m: laplacian_smooth(m, iterations=iterations, method=method)
        )
    except Exception as e:
        logger.error(f"Laplacian smooth failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/mesh-process/weld-vertices")
async def mesh_weld_vertices(
    mesh_file: UploadFile = File(...),
):
    """Merge duplicate vertices."""
    from trellis2.utils.mesh_processing import weld_vertices
    try:
        return await _load_and_process_mesh(mesh_file, weld_vertices)
    except Exception as e:
        logger.error(f"Weld vertices failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{request_id}/{file_name}")
async def download_file(request_id: str, file_name: str):
    file_path = os.path.join(OUTPUT_DIR, request_id, file_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, filename=file_name)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=52070, workers=1)
