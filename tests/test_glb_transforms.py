"""Tests for GLB transform sanitization logic."""
import sys
import os
import types
from unittest.mock import MagicMock

import numpy as np
import trimesh
import pytest

# ---------------------------------------------------------------------------
# Stub out heavy / GPU-only dependencies so trellis2_api can be imported in
# a plain test environment without CUDA, easydict, o_voxel, etc.
# ---------------------------------------------------------------------------

def _stub_module(name):
    """Insert a MagicMock module at sys.modules[name] (and all parents)."""
    parts = name.split(".")
    for i in range(1, len(parts) + 1):
        key = ".".join(parts[:i])
        if key not in sys.modules:
            sys.modules[key] = MagicMock()


_STUBS = [
    "torch",
    "PIL",
    "PIL.Image",
    "fastapi",
    "fastapi.responses",
    "fastapi.middleware",
    "fastapi.middleware.cors",
    "fastapi.concurrency",
    "trellis2",
    "trellis2.pipelines",
    "trellis2.pipelines.trellis2_texturing",
    "o_voxel",
    "easydict",
]

for _mod in _STUBS:
    _stub_module(_mod)

# FastAPI needs special treatment so FastAPI() and decorator calls don't fail
_fastapi_mock = sys.modules["fastapi"]
_fastapi_mock.FastAPI.return_value = MagicMock()

# Point OUTPUT_DIR to a writable temp directory so the module-level
# os.makedirs("/app/outputs") call doesn't fail on a dev/CI machine.
import tempfile as _tempfile
os.environ.setdefault("OUTPUT_DIR", _tempfile.mkdtemp(prefix="trellis2_test_"))

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Scene / transform helpers
# ---------------------------------------------------------------------------

def _make_scene(parts_config):
    """Helper: create a Scene from a list of (extents, transform_4x4) tuples."""
    scene = trimesh.Scene()
    for i, (extents, tf) in enumerate(parts_config):
        mesh = trimesh.creation.box(extents=extents)
        scene.add_geometry(mesh, node_name=f"part_{i}",
                           geom_name=f"geom_{i}", transform=tf)
    return scene


def _tf_translate(tx, ty, tz):
    tf = np.eye(4)
    tf[:3, 3] = [tx, ty, tz]
    return tf


def _tf_scale_translate(sx, sy, sz, tx, ty, tz):
    tf = np.eye(4)
    tf[0, 0], tf[1, 1], tf[2, 2] = sx, sy, sz
    tf[:3, 3] = [tx, ty, tz]
    return tf


def _tf_rotate_z_and_scale(angle_deg, scale, tx=0, ty=0, tz=0):
    a = np.radians(angle_deg)
    tf = np.eye(4)
    tf[:3, :3] = np.array([
        [np.cos(a), -np.sin(a), 0],
        [np.sin(a),  np.cos(a), 0],
        [0,          0,         1],
    ]) * scale
    tf[:3, 3] = [tx, ty, tz]
    return tf


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDetectSceneScale:

    def test_identity_transforms_return_1(self):
        from trellis2_api import _detect_scene_scale
        scene = _make_scene([
            ([2, 2, 2], _tf_translate(10, 0, 0)),
            ([1, 3, 1], _tf_translate(20, 0, 0)),
        ])
        assert _detect_scene_scale(scene) == pytest.approx(1.0, abs=0.01)

    def test_uniform_scale_detected(self):
        from trellis2_api import _detect_scene_scale
        scene = _make_scene([
            ([2, 2, 2], _tf_scale_translate(1000, 1000, 1000, 5000, 0, 0)),
            ([1, 3, 1], _tf_scale_translate(1000, 1000, 1000, 8000, 0, 0)),
        ])
        assert _detect_scene_scale(scene) == pytest.approx(1000.0, rel=0.01)

    def test_rotation_only_returns_1(self):
        from trellis2_api import _detect_scene_scale
        scene = _make_scene([
            ([2, 2, 2], _tf_rotate_z_and_scale(45, 1.0, tx=10)),
            ([1, 3, 1], _tf_rotate_z_and_scale(90, 1.0, tx=20)),
        ])
        assert _detect_scene_scale(scene) == pytest.approx(1.0, abs=0.01)

    def test_rotation_plus_scale(self):
        from trellis2_api import _detect_scene_scale
        scene = _make_scene([
            ([2, 2, 2], _tf_rotate_z_and_scale(45, 500.0, tx=10000)),
            ([1, 3, 1], _tf_rotate_z_and_scale(90, 500.0, tx=20000)),
        ])
        assert _detect_scene_scale(scene) == pytest.approx(500.0, rel=0.01)

    def test_single_mesh_with_scale(self):
        from trellis2_api import _detect_scene_scale
        scene = _make_scene([
            ([3, 5, 3], _tf_scale_translate(56000, 56000, 56000, 100, 50, 0)),
        ])
        assert _detect_scene_scale(scene) == pytest.approx(56000.0, rel=0.01)

    def test_empty_scene_returns_1(self):
        from trellis2_api import _detect_scene_scale
        scene = trimesh.Scene()
        assert _detect_scene_scale(scene) == pytest.approx(1.0)


import io


def _scene_to_glb_bytes(scene):
    """Export scene to in-memory GLB bytes."""
    return scene.export(file_type="glb")


def _glb_bytes_to_file(data, tmp_path, name="test.glb"):
    """Write GLB bytes to a temp file, return path."""
    p = tmp_path / name
    p.write_bytes(data)
    return str(p)


class TestLoadGlbParts:

    def test_no_scale_preserves_coords(self, tmp_path):
        """Parts without scale: world coords preserved as-is."""
        from trellis2_api import load_glb_parts
        scene = _make_scene([
            ([2, 2, 2], _tf_translate(10, 0, 0)),
            ([1, 3, 1], _tf_translate(20, 5, 0)),
        ])
        path = _glb_bytes_to_file(_scene_to_glb_bytes(scene), tmp_path)
        parts, merged, is_mp = load_glb_parts(path)
        assert is_mp is True
        assert len(parts) == 2
        assert merged.bounds[0][0] == pytest.approx(9.0, abs=0.1)
        assert merged.bounds[1][0] == pytest.approx(20.5, abs=0.1)

    def test_large_scale_stripped(self, tmp_path):
        """Parts with 1000x scale: output coords at local geometry scale."""
        from trellis2_api import load_glb_parts
        S = 1000.0
        scene = _make_scene([
            ([2, 2, 2], _tf_scale_translate(S, S, S, S * 10, 0, 0)),
            ([1, 3, 1], _tf_scale_translate(S, S, S, S * 20, S * 5, 0)),
        ])
        path = _glb_bytes_to_file(_scene_to_glb_bytes(scene), tmp_path)
        parts, merged, is_mp = load_glb_parts(path)
        assert is_mp is True
        bbox_range = merged.bounds[1] - merged.bounds[0]
        assert bbox_range.max() < 100

    def test_single_mesh_with_scale_stripped(self, tmp_path):
        """Single mesh + scale: scale still stripped from merged_ref."""
        from trellis2_api import load_glb_parts
        S = 56000.0
        scene = _make_scene([
            ([3, 5, 3], _tf_scale_translate(S, S, S, S * 2, S * 1, 0)),
        ])
        path = _glb_bytes_to_file(_scene_to_glb_bytes(scene), tmp_path)
        parts, merged, is_mp = load_glb_parts(path)
        assert is_mp is False
        bbox_range = merged.bounds[1] - merged.bounds[0]
        assert bbox_range.max() < 100

    def test_raw_trimesh_unchanged(self, tmp_path):
        """A plain .glb with a single Trimesh (no Scene): returned as-is."""
        from trellis2_api import load_glb_parts
        mesh = trimesh.creation.box(extents=[2, 3, 4])
        mesh.vertices += [10, 5, 0]
        path = _glb_bytes_to_file(mesh.export(file_type="glb"), tmp_path)
        parts, merged, is_mp = load_glb_parts(path)
        assert is_mp is False
        assert merged.bounds[1][0] == pytest.approx(11.0, abs=0.1)


class TestDenormalizeMesh:
    """Tests for _compute_normalization + _denormalize_mesh round-trip."""

    def test_round_trip(self):
        """normalize → denormalize should recover original bounds."""
        from trellis2_api import _compute_normalization, _denormalize_mesh
        mesh = trimesh.creation.box(extents=[4, 6, 2])
        mesh.vertices += [100, 50, 0]
        original_bounds = mesh.bounds.copy()

        center, scale = _compute_normalization(mesh)

        # Simulate pipeline output (axis swaps cancel, verts in normalized space)
        norm_verts = (mesh.vertices - center) * scale
        assert norm_verts.min() >= -0.5 - 1e-6
        assert norm_verts.max() <= 0.5 + 1e-6

        fake_output = trimesh.Trimesh(vertices=norm_verts, faces=mesh.faces, process=False)
        restored = _denormalize_mesh(fake_output, center, scale)

        np.testing.assert_allclose(restored.bounds, original_bounds, atol=0.01)

    def test_offset_mesh(self):
        """Mesh far from origin: denormalize restores position."""
        from trellis2_api import _compute_normalization, _denormalize_mesh
        mesh = trimesh.creation.box(extents=[2, 2, 2])
        mesh.vertices += [500, 300, 100]

        center, scale = _compute_normalization(mesh)
        norm_verts = (mesh.vertices - center) * scale
        fake_output = trimesh.Trimesh(vertices=norm_verts, faces=mesh.faces, process=False)
        restored = _denormalize_mesh(fake_output, center, scale)

        expected_center = np.array([500, 300, 100])
        actual_center = (restored.bounds[0] + restored.bounds[1]) / 2
        np.testing.assert_allclose(actual_center, expected_center, atol=0.01)
