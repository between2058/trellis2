"""
Mesh processing utilities: remeshing, simplification, hole filling, smoothing.

These functions wrap CuMesh, Meshlib (mrmeshpy), Open3D, and trimesh
to provide a clean API for mesh post-processing operations.
"""
from typing import Optional, Literal
import numpy as np
import torch
import trimesh


def simplify_mesh(
    mesh: trimesh.Trimesh,
    target_face_num: int,
    method: Literal['cumesh', 'meshlib'] = 'cumesh',
) -> trimesh.Trimesh:
    """Reduce face count of a mesh."""
    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()

    if method == 'cumesh':
        import cumesh
        cm = cumesh.CuMesh()
        cm.init(vertices, faces)
        cm.simplify(target_face_num, verbose=True)
        v_out, f_out = cm.read()
    else:
        import meshlib.mrmeshpy as mrmeshpy
        import meshlib.mrmeshnumpy as mrmeshnumpy
        faces_np = mesh.faces.astype(np.int32)
        verts_np = mesh.vertices.astype(np.float32)
        mr_mesh = mrmeshnumpy.meshFromFacesVerts(faces_np, verts_np)
        mr_mesh.packOptimally()
        settings = mrmeshpy.DecimateSettings()
        settings.maxDeletedFaces = max(0, len(mesh.faces) - target_face_num)
        settings.packMesh = True
        mrmeshpy.decimateMesh(mr_mesh, settings)
        v_out = torch.from_numpy(mrmeshnumpy.getNumpyVerts(mr_mesh)).float()
        f_out = torch.from_numpy(mrmeshnumpy.getNumpyFaces(mr_mesh.topology)).int()

    return trimesh.Trimesh(
        vertices=v_out.cpu().numpy(),
        faces=f_out.cpu().numpy(),
        process=False,
    )


def remesh_dual_contouring(
    mesh: trimesh.Trimesh,
    resolution: int = 512,
    band: float = 1.0,
    project_back: float = 0.0,
    remove_floaters: bool = True,
    remove_inner_faces: bool = True,
) -> trimesh.Trimesh:
    """Rebuild mesh topology using dual contouring."""
    import cumesh

    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    faces = torch.from_numpy(mesh.faces).int().cuda()

    vmin = vertices.min(dim=0).values
    vmax = vertices.max(dim=0).values
    center = (vmin + vmax) / 2
    scale = (vmax - vmin).max()

    v_out, f_out = cumesh.remeshing.remesh_narrow_band_dc(
        vertices, faces,
        center=center.tolist(),
        scale=scale.item(),
        resolution=resolution,
        band=band,
        project_back=project_back,
        remove_inner_faces=remove_inner_faces,
    )

    out = trimesh.Trimesh(vertices=v_out.cpu().numpy(), faces=f_out.cpu().numpy(), process=False)

    if remove_floaters:
        out = _remove_floaters(out)

    return out


def fill_holes(mesh: trimesh.Trimesh, method: Literal['meshlib', 'cumesh'] = 'meshlib', max_perimeter: float = 0.03) -> trimesh.Trimesh:
    """Fill holes in mesh."""
    if method == 'meshlib':
        import meshlib.mrmeshpy as mrmeshpy
        import meshlib.mrmeshnumpy as mrmeshnumpy
        faces_np = mesh.faces.astype(np.int32)
        verts_np = mesh.vertices.astype(np.float32)
        mr_mesh = mrmeshnumpy.meshFromFacesVerts(faces_np, verts_np)
        hole_edges = mr_mesh.topology.findHoleRepresentiveEdges()
        for edge in hole_edges:
            params = mrmeshpy.FillHoleParams()
            params.metric = mrmeshpy.getUniversalMetric(mr_mesh)
            mrmeshpy.fillHole(mr_mesh, edge, params)
        v_out = mrmeshnumpy.getNumpyVerts(mr_mesh)
        f_out = mrmeshnumpy.getNumpyFaces(mr_mesh.topology)
        return trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
    else:
        import cumesh
        vertices = torch.from_numpy(mesh.vertices).float().cuda()
        faces = torch.from_numpy(mesh.faces).int().cuda()
        cm = cumesh.CuMesh()
        cm.init(vertices, faces)
        cm.fill_holes(max_hole_perimeter=max_perimeter)
        v_out, f_out = cm.read()
        return trimesh.Trimesh(
            vertices=v_out.cpu().numpy(),
            faces=f_out.cpu().numpy(),
            process=False,
        )


def smooth_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Compute smooth vertex normals (area-weighted)."""
    new_mesh = mesh.copy()
    # trimesh auto-computes area-weighted vertex normals; force recalculation
    new_mesh.vertex_normals  # noqa: triggers computation
    return new_mesh


def laplacian_smooth(
    mesh: trimesh.Trimesh,
    iterations: int = 5,
    method: Literal['laplacian', 'taubin'] = 'laplacian',
) -> trimesh.Trimesh:
    """Smooth mesh geometry using Laplacian or Taubin smoothing."""
    import open3d as o3d

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)

    if method == 'laplacian':
        o3d_mesh = o3d_mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    else:
        o3d_mesh = o3d_mesh.filter_smooth_taubin(number_of_iterations=iterations)

    return trimesh.Trimesh(
        vertices=np.asarray(o3d_mesh.vertices),
        faces=np.asarray(o3d_mesh.triangles),
        process=False,
    )


def weld_vertices(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Merge duplicate vertices."""
    mesh = mesh.copy()
    mesh.merge_vertices()
    return mesh


def _remove_floaters(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Remove disconnected components, keeping largest."""
    components = mesh.split(only_watertight=False)
    if len(components) <= 1:
        return mesh
    largest = max(components, key=lambda c: len(c.faces))
    return largest
