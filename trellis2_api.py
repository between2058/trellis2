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
# Multi-part GLB helpers
# =============================================================================

def load_glb_parts(path: str):
    """Load GLB and return (parts, merged_mesh, is_multipart).

    Uses force='mesh' as ground truth for the merged mesh (handles all GLB
    structures correctly). Extracts individual parts from the scene graph
    with world transforms baked in. If the parts don't match force='mesh',
    falls back to single-mesh mode.
    """
    loaded = _trimesh.load(path, process=False)

    if isinstance(loaded, _trimesh.Trimesh):
        return [loaded], loaded, False

    if not isinstance(loaded, _trimesh.Scene):
        raise ValueError(f"Unexpected type from trimesh.load: {type(loaded)}")

    # Ground truth: force='mesh' correctly handles ALL transform structures
    merged_ref = _trimesh.load(path, force='mesh', process=False)
    ref_bbox = merged_ref.bounds  # (2, 3) array: [min, max]
    logger.info(f"force='mesh' reference: {len(merged_ref.vertices)} verts, "
                f"bbox_range={ref_bbox[1] - ref_bbox[0]}")

    # Extract parts with world transforms from scene graph
    node_meshes = {}
    for node_name in loaded.graph.nodes_geometry:
        transform, geom_name = loaded.graph[node_name]
        geom = loaded.geometry[geom_name]
        if not isinstance(geom, _trimesh.Trimesh) or len(geom.faces) == 0:
            continue

        local_center = (geom.bounds[0] + geom.bounds[1]) / 2
        translation = transform[:3, 3]
        is_identity = np.allclose(transform, np.eye(4), atol=1e-6)
        logger.debug(f"  Node '{node_name}': local_center={local_center.tolist()}, "
                     f"translation={translation.tolist()}, identity={is_identity}")

        mesh = geom.copy().apply_transform(transform)
        node_meshes.setdefault(node_name, []).append(mesh)

    if not node_meshes:
        return [merged_ref], merged_ref, False

    parts = []
    for meshes in node_meshes.values():
        part = meshes[0] if len(meshes) == 1 else _trimesh.util.concatenate(meshes)
        if len(part.faces) > 0:
            parts.append(part)

    if len(parts) <= 1:
        return [merged_ref], merged_ref, False

    # Verify: our concatenated parts should match force='mesh' bounding box
    concat = _trimesh.util.concatenate(parts)
    concat_bbox = concat.bounds
    bbox_match = np.allclose(ref_bbox, concat_bbox, atol=0.01)
    logger.info(f"Extracted {len(parts)} parts, concat {len(concat.vertices)} verts, "
                f"bbox_range={concat_bbox[1] - concat_bbox[0]}, "
                f"match_ref={bbox_match}")

    if not bbox_match:
        logger.warning(
            f"Part extraction bbox mismatch! "
            f"ref={ref_bbox.tolist()} vs ours={concat_bbox.tolist()}. "
            f"Retrying with scene.dump()..."
        )
        # scene.dump applies transforms the same way force='mesh' does internally
        dumped = loaded.dump(concatenate=False)
        parts = [m for m in dumped
                 if isinstance(m, _trimesh.Trimesh) and len(m.faces) > 0]
        if len(parts) <= 1:
            return [merged_ref], merged_ref, False
        concat2 = _trimesh.util.concatenate(parts)
        bbox_match2 = np.allclose(ref_bbox, concat2.bounds, atol=0.01)
        logger.info(f"scene.dump() gave {len(parts)} parts, match_ref={bbox_match2}")
        if not bbox_match2:
            logger.warning("Still mismatched — falling back to single mesh")
            return [merged_ref], merged_ref, False

    # Always use force='mesh' for the pipeline input (guaranteed correct)
    return parts, merged_ref, True


def _extract_submesh(textured, face_mask):
    """Extract a sub-mesh from textured mesh using a boolean face mask."""
    faces = textured.faces[face_mask]
    if len(faces) == 0:
        return None

    uv_idx, remap = np.unique(faces.ravel(), return_inverse=True)
    new_faces = remap.reshape(-1, 3)
    new_verts = textured.vertices[uv_idx]

    kwargs = dict(vertices=new_verts, faces=new_faces, process=False)

    try:
        kwargs['vertex_normals'] = textured.vertex_normals[uv_idx]
    except Exception:
        pass

    has_uv = (isinstance(textured.visual, _trimesh.visual.TextureVisuals)
              and textured.visual.uv is not None)
    if has_uv:
        kwargs['visual'] = _trimesh.visual.TextureVisuals(
            uv=textured.visual.uv[uv_idx],
            material=textured.visual.material)

    return _trimesh.Trimesh(**kwargs)


def split_textured_mesh(textured, parts, center, scale):
    """Split textured mesh back into original parts using vertex position matching.

    After pipeline.run():
      - preprocess_mesh normalizes: v_norm = (v_orig - center) * scale, then axis swap
      - postprocess_mesh UV unwraps (may duplicate/remove vertices), then undoes axis swap
      - Result vertices are at: (v_orig - center) * scale  (axis swaps cancel out)

    We match each textured vertex to the closest original part vertex (also in
    normalized space) via KD-tree, then group faces by part assignment.
    """
    from scipy.spatial import cKDTree

    # Build normalized vertex positions for each part + a part-ID label per vertex
    all_norm = []
    all_part_id = []
    for i, part in enumerate(parts):
        norm = (part.vertices - center) * scale
        all_norm.append(norm)
        all_part_id.append(np.full(len(norm), i, dtype=np.int32))

    all_norm = np.vstack(all_norm).astype(np.float64)
    all_part_id = np.concatenate(all_part_id)

    tree = cKDTree(all_norm)

    # Textured vertices are already in normalized space (axis swaps cancelled)
    dists, indices = tree.query(textured.vertices.astype(np.float64))
    max_dist = dists.max()
    logger.info(f"Vertex matching: max_dist={max_dist:.8f}, mean_dist={dists.mean():.8f}")

    if max_dist > 0.01:
        logger.warning(f"Spatial matching too imprecise (max_dist={max_dist:.4f}), giving up split")
        return None

    # Each textured vertex → part ID
    vert_part = all_part_id[indices]

    # Each face → part ID (use first vertex; all 3 should agree)
    face_part = vert_part[textured.faces[:, 0]]

    n_parts = len(parts)
    result = []
    for i in range(n_parts):
        mask = face_part == i
        n_faces = mask.sum()
        if n_faces == 0:
            logger.warning(f"Part {i}: 0 faces after matching, skipping")
            continue
        logger.info(f"Part {i}: {n_faces} faces (original {len(parts[i].faces)})")
        sub = _extract_submesh(textured, mask)
        if sub is not None:
            result.append(sub)

    return result if len(result) > 0 else None


def _split_and_assemble(textured, parts, center, scale, debug_dir=None):
    """Split textured mesh into parts, restore world coordinates, assemble Scene.

    Pipeline output is in normalized space [-0.5, 0.5]. We restore each part's
    vertices to original world coordinates: v_world = v_normalized / scale + center.
    This ensures the output GLB has the same coordinate system as the input.

    Returns a trimesh.Scene (multipart) or trimesh.Trimesh (fallback).
    """
    result_parts = split_textured_mesh(textured, parts, center, scale)

    if result_parts is None:
        logger.warning("Split failed — returning single textured mesh")
        return textured

    # Restore world coordinates: reverse the normalization
    # Pipeline did: v_norm = (v_orig - center) * scale
    # Restore:      v_orig = v_norm / scale + center
    for i, part in enumerate(result_parts):
        part.vertices = part.vertices / scale + center
        p_min = part.vertices.min(0)
        p_max = part.vertices.max(0)
        p_center = (p_min + p_max) / 2
        logger.info(f"[DIAG] Restored part {i}: {len(part.vertices)} verts, {len(part.faces)} faces, "
                    f"bbox=[{p_min.tolist()}, {p_max.tolist()}], center={p_center.tolist()}")

    scene = _trimesh.Scene()
    for i, part in enumerate(result_parts):
        scene.add_geometry(part, node_name=f"part_{i}")

    if scene.bounds is not None:
        logger.info(f"[DIAG] Scene bounds: {scene.bounds.tolist()}")

    return scene


def texture_multipart(pipe, parts, merged, image, seed=42,
                      resolution=1024, texture_size=2048, debug_dir=None):
    """Texture multi-part GLB: merge → run pipeline once → split."""
    logger.info(f"Multipart texture: {len(parts)} parts, "
                f"merged={len(merged.faces)} faces")

    # center/scale needed for KD-tree matching (not for coord restoration)
    v = merged.vertices
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()

    # Debug: verify merged mesh vertex positions and export
    if debug_dir:
        try:
            logger.info(f"[DIAG] Merged mesh in memory: {len(merged.vertices)} verts, "
                        f"bbox=[{vmin.tolist()}, {vmax.tolist()}]")
            logger.info(f"[DIAG] Merged first 3 verts: {merged.vertices[:3].tolist()}")
            logger.info(f"[DIAG] Merged last 3 verts: {merged.vertices[-3:].tolist()}")
            logger.info(f"[DIAG] Merged vertex dtype: {merged.vertices.dtype}")

            # Export as GLB
            merged.export(os.path.join(debug_dir, "debug_merged.glb"))

            # Readback GLB to check if export preserved coordinates
            rb = _trimesh.load(os.path.join(debug_dir, "debug_merged.glb"), force='mesh', process=False)
            logger.info(f"[DIAG] Readback debug_merged.glb: {len(rb.vertices)} verts, "
                        f"bbox=[{rb.vertices.min(0).tolist()}, {rb.vertices.max(0).tolist()}]")
            logger.info(f"[DIAG] Readback first 3 verts: {rb.vertices[:3].tolist()}")

            # Also export as OBJ (text format, can verify manually)
            merged.export(os.path.join(debug_dir, "debug_merged.obj"))
            logger.info(f"[DIAG] Wrote debug_merged.obj for manual verification")
        except Exception as e:
            logger.warning(f"[DIAG] debug_merged export failed: {e}")

    textured = pipe.run(merged, image, seed=seed,
                        resolution=resolution, texture_size=texture_size)
    logger.info(f"Pipeline output: {len(textured.faces)} faces, "
                f"{len(textured.vertices)} verts")

    # Debug: save pipeline output as single mesh (no splitting) for comparison
    if debug_dir:
        try:
            debug_single = textured.copy()
            debug_single.vertices = debug_single.vertices / scale + center
            debug_single.export(os.path.join(debug_dir, "debug_single.glb"))
            logger.info("[DIAG] Wrote debug_single.glb (pipeline output, restored coords, no split)")
        except Exception as e:
            logger.warning(f"[DIAG] Failed to write debug_single: {e}")

    return _split_and_assemble(textured, parts, center, scale, debug_dir=debug_dir)


def texture_multipart_multiview(pipe, parts, merged,
                                front_image, back_image=None,
                                left_image=None, right_image=None,
                                seed=42, resolution=1024, texture_size=2048,
                                front_axis='z', blend_temperature=2.0,
                                debug_dir=None):
    """Multi-view version of texture_multipart."""
    logger.info(f"Multipart texture (multiview): {len(parts)} parts, "
                f"merged={len(merged.faces)} faces")

    v = merged.vertices
    vmin, vmax = v.min(axis=0), v.max(axis=0)
    center = (vmin + vmax) / 2
    scale = 0.99999 / (vmax - vmin).max()

    textured = pipe.run_multiview(
        merged, front_image=front_image, back_image=back_image,
        left_image=left_image, right_image=right_image,
        seed=seed, resolution=resolution, texture_size=texture_size,
        front_axis=front_axis, blend_temperature=blend_temperature,
    )
    logger.info(f"Pipeline output: {len(textured.faces)} faces, "
                f"{len(textured.vertices)} verts")

    return _split_and_assemble(textured, parts, center, scale, debug_dir=debug_dir)


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
                    result = texture_multipart(
                        tex_pipeline, parts, merged, image,
                        seed=seed, resolution=resolution, texture_size=texture_size,
                        debug_dir=req_dir,
                    )
                else:
                    result = tex_pipeline.run(
                        merged, image, seed=seed,
                        resolution=resolution, texture_size=texture_size,
                    )
                result.export(out_path)

                # Read back exported GLB to verify part positions
                if is_multipart:
                    try:
                        readback = _trimesh.load(out_path, process=False)
                        if isinstance(readback, _trimesh.Scene):
                            for nn in readback.graph.nodes_geometry:
                                tf, gn = readback.graph[nn]
                                g = readback.geometry[gn]
                                rb_min, rb_max = g.vertices.min(0), g.vertices.max(0)
                                rb_t = tf[:3, 3]
                                logger.info(f"[DIAG] Readback node '{nn}': "
                                            f"geom_bbox=[{rb_min.tolist()}, {rb_max.tolist()}], "
                                            f"translation={rb_t.tolist()}")
                        else:
                            logger.info(f"[DIAG] Readback is single Trimesh, "
                                        f"bbox=[{readback.vertices.min(0).tolist()}, "
                                        f"{readback.vertices.max(0).tolist()}]")
                    except Exception as e:
                        logger.warning(f"[DIAG] Readback failed: {e}")

                return out_path

            out_path = await run_in_threadpool(_run)
            log_gpu_memory("after texture")
            torch.cuda.empty_cache()

        return {
            "request_id": request_id,
            "glb_url": f"/download/{request_id}/output.glb",
            "debug_concat_url": f"/download/{request_id}/debug_concat.glb" if is_multipart else None,
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
                    result = texture_multipart_multiview(
                        tex_pipeline, parts, merged,
                        front_image=front_img, back_image=back_img,
                        left_image=left_img, right_image=right_img,
                        seed=seed, resolution=resolution, texture_size=texture_size,
                        front_axis=front_axis, blend_temperature=blend_temperature,
                        debug_dir=req_dir,
                    )
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
