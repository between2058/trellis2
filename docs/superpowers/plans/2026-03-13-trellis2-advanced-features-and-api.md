# TRELLIS.2 Advanced Features & FastAPI Service Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend TRELLIS.2 with multi-view generation, mesh processing utilities, advanced samplers, and serve it all as a Dockerized FastAPI service.

**Architecture:** Add multi-view spatial blending at the sampler level, expose multiview pipeline methods on existing pipeline classes, add mesh processing utilities, then wrap everything in a FastAPI server following ReconViaGen patterns.

**Tech Stack:** PyTorch, FastAPI, uvicorn, CuMesh, Meshlib (mrmeshpy), Open3D, trimesh, Docker (nvidia/cuda base)

---

## Chunk 1: Multi-View Samplers & Advanced Samplers

### Task 1: Add Multi-View and Higher-Order Samplers

**Files:**
- Modify: `trellis2/pipelines/samplers/flow_euler.py` — add all new sampler classes
- Modify: `trellis2/pipelines/samplers/__init__.py` — export new classes

We add all sampler classes directly into `flow_euler.py` following the existing pattern (mixins + composition). This matches what ComfyUI-Trellis2 does and keeps all sampler logic in one place.

- [ ] **Step 1: Add FlowEulerMultiViewSampler to flow_euler.py**

Append to `trellis2/pipelines/samplers/flow_euler.py` after the existing `FlowEulerGuidanceIntervalSampler` class:

```python
class FlowEulerMultiViewSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with multi-view blending.
    """
    def __init__(self, sigma_min: float, resolution: int):
        super().__init__(sigma_min)
        self.resolution = resolution

    def _compute_view_weights_sparse(self, coords, views, front_axis='z', blend_temperature=2.0) -> torch.Tensor:
        z = (coords[:, 1].float() / self.resolution) * 2 - 1.0
        x = (coords[:, 3].float() / self.resolution) * 2 - 1.0

        if front_axis == 'z':
            view_vectors = {
                'front': torch.stack([torch.zeros_like(z), z], dim=1),
                'back':  torch.stack([torch.zeros_like(z), -z], dim=1),
                'right': torch.stack([x, torch.zeros_like(x)], dim=1),
                'left':  torch.stack([-x, torch.zeros_like(x)], dim=1),
            }
        else:
            view_vectors = {
                'front': torch.stack([x, torch.zeros_like(x)], dim=1),
                'back':  torch.stack([-x, torch.zeros_like(x)], dim=1),
                'right': torch.stack([torch.zeros_like(z), z], dim=1),
                'left':  torch.stack([torch.zeros_like(z), -z], dim=1),
            }

        scores = []
        for view in views:
            if view in view_vectors:
                scores.append(view_vectors[view].sum(dim=1))
            else:
                scores.append(torch.full_like(z, -10.0))

        scores = torch.stack(scores, dim=1)
        return torch.softmax(scores * blend_temperature, dim=1)

    def _compute_view_weights_dense(self, shape, device, views, front_axis='z', blend_temperature=2.0) -> torch.Tensor:
        D, H, W = shape[2], shape[3], shape[4]
        dz = torch.linspace(-1, 1, D, device=device)
        dy = torch.linspace(-1, 1, H, device=device)
        dx = torch.linspace(-1, 1, W, device=device)
        grid_z, grid_y, grid_x = torch.meshgrid(dz, dy, dx, indexing='ij')

        if front_axis == 'z':
            view_scores = {'front': grid_z, 'back': -grid_z, 'right': grid_x, 'left': -grid_x}
        else:
            view_scores = {'front': grid_x, 'back': -grid_x, 'right': grid_z, 'left': -grid_z}

        scores = []
        for view in views:
            scores.append(view_scores.get(view, torch.full_like(grid_z, -10.0)))
        scores = torch.stack(scores, dim=0)
        return torch.softmax(scores * blend_temperature, dim=0)

    @torch.no_grad()
    def sample_once(
        self, model, x_t, t: float, t_prev: float,
        conds: Dict[str, Any], views: List[str],
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        is_sparse = hasattr(x_t, 'coords')

        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)

        pred_v_accum = 0
        for i, view in enumerate(views):
            cond = conds[view]
            if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                pred_v_view = self._inference_model(model, x_t, t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
            else:
                pred_v_view = self._inference_model(model, x_t, t, cond=cond, **kwargs)

            if is_sparse:
                w = weights[:, i].unsqueeze(1)
                v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                pred_v_accum = pred_v_accum + v_feats * w
            else:
                w = weights[i].unsqueeze(0).unsqueeze(0)
                pred_v_accum = pred_v_accum + pred_v_view * w

        if is_sparse:
            pred_v = x_t.replace(feats=pred_v_accum)
        else:
            pred_v = pred_v_accum
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self, model, noise, conds: Dict[str, Any], views: List[str],
        steps: int = 50, rescale_t: float = 1.0, verbose: bool = True,
        tqdm_desc: str = "Sampling MultiView",
        front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs
    ):
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(
                model, sample, t, t_prev, conds=conds, views=views,
                front_axis=front_axis, blend_temperature=blend_temperature, **kwargs
            )
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerMultiViewGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerMultiViewSampler):
    """Multi-view Euler with CFG and guidance interval."""
    pass
```

- [ ] **Step 2: Add RK4 sampler (single-view and multi-view)**

Append to `trellis2/pipelines/samplers/flow_euler.py`:

```python
class FlowRK4Sampler(FlowEulerSampler):
    """4th-order Runge-Kutta flow matching sampler."""
    @torch.no_grad()
    def sample_once(self, model, x_t, t: float, t_prev: float, cond: Optional[Any] = None, **kwargs):
        dt = t_prev - t
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v
        k1 = get_v(x_t, t)
        k2 = get_v(x_t + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = get_v(x_t + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = get_v(x_t + dt * k3, t + dt)
        pred_x_prev = x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK4CfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowRK4Sampler):
    """RK4 with classifier-free guidance."""
    @torch.no_grad()
    def sample(self, model, noise, cond, neg_cond, steps=50, rescale_t=1.0, guidance_strength=3.0, verbose=True, **kwargs):
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowRK4GuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK4Sampler):
    """RK4 with CFG and guidance interval."""
    pass


class FlowRK4MultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view RK4 sampler."""
    @torch.no_grad()
    def sample_once(self, model, x_t, t: float, t_prev: float,
                    conds: Dict[str, Any], views: List[str],
                    front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)

        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                if is_sparse:
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum = pred_v_accum + v_feats * weights[:, i].unsqueeze(1)
                else:
                    pred_v_accum = pred_v_accum + pred_v_view * weights[i].unsqueeze(0).unsqueeze(0)
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            return pred_v_accum

        k1 = get_blended_v(x_t, t)
        k2 = get_blended_v(x_t + k1 * (0.5 * dt), t + 0.5 * dt)
        k3 = get_blended_v(x_t + k2 * (0.5 * dt), t + 0.5 * dt)
        k4 = get_blended_v(x_t + k3 * dt, t + dt)
        pred_x_prev = x_t + (k1 + k2 * 2 + k3 * 2 + k4) * (dt / 6.0)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowRK4MultiViewGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowRK4MultiViewSampler):
    pass
```

- [ ] **Step 3: Add Heun sampler (single-view and multi-view)**

Append to `trellis2/pipelines/samplers/flow_euler.py`:

```python
class FlowHeunSampler(FlowEulerSampler):
    """Heun's method (2nd-order Runge-Kutta) flow matching sampler."""
    @torch.no_grad()
    def sample_once(self, model, x_t, t: float, t_prev: float, cond: Optional[Any] = None, **kwargs):
        dt = t_prev - t
        def get_v(current_x, current_t):
            _, _, pred_v = self._get_model_prediction(model, current_x, current_t, cond, **kwargs)
            return pred_v
        k1 = get_v(x_t, t)
        x_temp = x_t + k1 * dt
        k2 = get_v(x_temp, t + dt)
        pred_x_prev = x_t + 0.5 * dt * (k1 + k2)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowHeunGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowHeunSampler):
    """Heun with CFG and guidance interval."""
    pass


class FlowHeunMultiViewSampler(FlowEulerMultiViewSampler):
    """Multi-view Heun sampler."""
    @torch.no_grad()
    def sample_once(self, model, x_t, t: float, t_prev: float,
                    conds: Dict[str, Any], views: List[str],
                    front_axis: str = 'z', blend_temperature: float = 2.0, **kwargs):
        dt = t_prev - t
        is_sparse = hasattr(x_t, 'coords')
        if is_sparse:
            weights = self._compute_view_weights_sparse(x_t.coords, views, front_axis, blend_temperature)
        else:
            weights = self._compute_view_weights_dense(x_t.shape, x_t.device, views, front_axis, blend_temperature)

        def get_blended_v(current_x, current_t):
            pred_v_accum = 0
            for i, view in enumerate(views):
                cond = conds[view]
                if isinstance(cond, dict) and 'cond' in cond and 'neg_cond' in cond:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond['cond'], neg_cond=cond['neg_cond'], **kwargs)
                else:
                    pred_v_view = self._inference_model(model, current_x, current_t, cond=cond, **kwargs)
                if is_sparse:
                    v_feats = pred_v_view.feats if hasattr(pred_v_view, 'feats') else pred_v_view
                    pred_v_accum = pred_v_accum + v_feats * weights[:, i].unsqueeze(1)
                else:
                    pred_v_accum = pred_v_accum + pred_v_view * weights[i].unsqueeze(0).unsqueeze(0)
            if is_sparse:
                return current_x.replace(feats=pred_v_accum)
            return pred_v_accum

        k1 = get_blended_v(x_t, t)
        x_temp = x_t + k1 * dt
        k2 = get_blended_v(x_temp, t + dt)
        pred_x_prev = x_t + 0.5 * dt * (k1 + k2)
        pred_x_0, _ = self._v_to_xstart_eps(x_t=x_t, t=t, v=k1)
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})


class FlowHeunMultiViewGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowHeunMultiViewSampler):
    pass
```

- [ ] **Step 4: Update `__init__.py` exports**

Replace `trellis2/pipelines/samplers/__init__.py` with:

```python
from .base import Sampler
from .flow_euler import (
    FlowEulerSampler,
    FlowEulerCfgSampler,
    FlowEulerGuidanceIntervalSampler,
    FlowEulerMultiViewSampler,
    FlowEulerMultiViewGuidanceIntervalSampler,
    FlowRK4Sampler,
    FlowRK4CfgSampler,
    FlowRK4GuidanceIntervalSampler,
    FlowRK4MultiViewSampler,
    FlowRK4MultiViewGuidanceIntervalSampler,
    FlowHeunSampler,
    FlowHeunGuidanceIntervalSampler,
    FlowHeunMultiViewSampler,
    FlowHeunMultiViewGuidanceIntervalSampler,
)
```

- [ ] **Step 5: Verify imports work**

Run: `cd /Users/between2058/Documents/code/TRELLIS.2 && python -c "from trellis2.pipelines.samplers import FlowEulerMultiViewSampler, FlowRK4Sampler, FlowHeunSampler; print('All samplers imported OK')"`

- [ ] **Step 6: Commit**

```bash
git add trellis2/pipelines/samplers/flow_euler.py trellis2/pipelines/samplers/__init__.py
git commit -m "feat: add multi-view, RK4, and Heun samplers for flow matching"
```

---

## Chunk 2: Multi-View Pipeline Methods

### Task 2: Add multi-view methods to image-to-3D pipeline

**Files:**
- Modify: `trellis2/pipelines/trellis2_image_to_3d.py` — add `run_multiview`, `sample_*_multiview` methods, `switch_samplers`, helper methods

- [ ] **Step 1: Add imports and helpers at top of file**

Add after existing imports in `trellis2/pipelines/trellis2_image_to_3d.py`:

```python
import gc
import random

def seed_all(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

- [ ] **Step 2: Add helper methods to `Trellis2ImageTo3DPipeline`**

Add these methods to the class (after `__init__`, before `from_pretrained`):

```python
    def switch_samplers(self, sampler_type: str = "euler"):
        """Dynamically switch sampler instances based on type."""
        prefix = {"rk4": "RK4", "heun": "Heun"}.get(sampler_type, "Euler")
        self._sampler_prefix = prefix
        args = self._pretrained_args
        self.sparse_structure_sampler = getattr(samplers, f"Flow{prefix}GuidanceIntervalSampler")(**args['sparse_structure_sampler']['args'])
        self.shape_slat_sampler = getattr(samplers, f"Flow{prefix}GuidanceIntervalSampler")(**args['shape_slat_sampler']['args'])
        self.tex_slat_sampler = getattr(samplers, f"Flow{prefix}GuidanceIntervalSampler")(**args['tex_slat_sampler']['args'])

    def _cond_to(self, cond: dict, device: torch.device) -> dict:
        return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in cond.items()}

    def _cond_cpu(self, cond: dict) -> dict:
        return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in cond.items()}

    def _cleanup_cuda(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
```

- [ ] **Step 3: Add `sample_sparse_structure_multiview` method**

```python
    def sample_sparse_structure_multiview(
        self, conds: dict, views: list, resolution: int,
        num_samples: int = 1, sampler_params: dict = {},
        front_axis: str = 'z', blend_temperature: float = 2.0,
    ) -> torch.Tensor:
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)
        flow_model = self.models['sparse_structure_flow_model']
        reso = flow_model.resolution
        noise = torch.randn(num_samples, flow_model.in_channels, reso, reso, reso).to(self.device)

        sampler_class = getattr(samplers, f"Flow{self._sampler_prefix}MultiViewGuidanceIntervalSampler",
                                samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(sigma_min=self.sparse_structure_sampler.sigma_min, resolution=reso)
        sampler_params = {**self.sparse_structure_sampler_params, **sampler_params}

        if self.low_vram:
            flow_model.to(self.device)
        z_s = sampler.sample(
            flow_model, noise, conds=conds, views=views,
            front_axis=front_axis, blend_temperature=blend_temperature,
            **sampler_params, verbose=True, tqdm_desc="Sampling sparse structure (MultiView)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()

        decoder = self.models['sparse_structure_decoder']
        if self.low_vram:
            decoder.to(self.device)
        decoded = decoder(z_s) > 0
        if self.low_vram:
            decoder.cpu()
            self._cleanup_cuda()

        if resolution != decoded.shape[2]:
            if resolution < decoded.shape[2]:
                ratio = decoded.shape[2] // resolution
                decoded = torch.nn.functional.max_pool3d(decoded.float(), ratio, ratio, 0) > 0.5
            else:
                decoded = torch.nn.functional.interpolate(decoded.float(), size=(resolution,)*3, mode='nearest') > 0.5

        coords = torch.argwhere(decoded)[:, [0, 2, 3, 4]].int()
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()
        return coords
```

- [ ] **Step 4: Add `sample_shape_slat_multiview` method**

```python
    def sample_shape_slat_multiview(
        self, conds: dict, views: list, flow_model,
        coords: torch.Tensor, sampler_params: dict = {},
        front_axis: str = 'z', blend_temperature: float = 2.0,
    ) -> SparseTensor:
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)
        coords_dev = coords.to(self.device)
        noise = SparseTensor(
            feats=torch.randn(coords.shape[0], flow_model.in_channels, device=self.device),
            coords=coords_dev,
        )
        sampler_class = getattr(samplers, f"Flow{self._sampler_prefix}MultiViewGuidanceIntervalSampler",
                                samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(sigma_min=self.shape_slat_sampler.sigma_min, resolution=flow_model.resolution)
        sampler_params = {**self.shape_slat_sampler_params, **sampler_params}

        if self.low_vram:
            flow_model.to(self.device)
        slat = sampler.sample(
            flow_model, noise, conds=conds, views=views,
            front_axis=front_axis, blend_temperature=blend_temperature,
            **sampler_params, verbose=True, tqdm_desc="Sampling shape SLat (MultiView)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()
        return slat
```

- [ ] **Step 5: Add `sample_tex_slat_multiview` method**

```python
    def sample_tex_slat_multiview(
        self, conds: dict, views: list, shape_slat: SparseTensor,
        flow_model, sampler_params: dict = {},
        front_axis: str = 'z', blend_temperature: float = 2.0,
    ) -> SparseTensor:
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_to(conds[v], self.device)

        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat_normalized = (shape_slat - mean) / std

        in_channels = flow_model.in_channels if isinstance(flow_model, nn.Module) else flow_model[0].in_channels
        noise = shape_slat.replace(
            feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device)
        )
        sampler_params = {**self.tex_slat_sampler_params, **sampler_params}

        sampler_class = getattr(samplers, f"Flow{self._sampler_prefix}MultiViewGuidanceIntervalSampler",
                                samplers.FlowEulerMultiViewGuidanceIntervalSampler)
        sampler = sampler_class(
            sigma_min=self.tex_slat_sampler.sigma_min,
            resolution=flow_model.resolution if hasattr(flow_model, 'resolution') else flow_model[0].resolution
        )

        if self.low_vram:
            flow_model.to(self.device)
        slat = sampler.sample(
            flow_model, noise, conds=conds, views=views,
            front_axis=front_axis, blend_temperature=blend_temperature,
            concat_cond=shape_slat_normalized,
            **sampler_params, verbose=True, tqdm_desc="Sampling texture SLat (MultiView)",
        ).samples
        if self.low_vram:
            flow_model.cpu()
            self._cleanup_cuda()

        std = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std + mean
        if self.low_vram:
            for v in conds:
                conds[v] = self._cond_cpu(conds[v])
            self._cleanup_cuda()
        return slat
```

- [ ] **Step 6: Add `run_multiview` method**

```python
    @torch.no_grad()
    def run_multiview(
        self,
        front_image: Image.Image,
        back_image: Optional[Image.Image] = None,
        left_image: Optional[Image.Image] = None,
        right_image: Optional[Image.Image] = None,
        num_samples: int = 1,
        seed: int = 42,
        sparse_structure_sampler_params: dict = {},
        shape_slat_sampler_params: dict = {},
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        pipeline_type: Optional[str] = None,
        max_num_tokens: int = 49152,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
        sampler: str = 'euler',
    ) -> List[MeshWithVoxel]:
        self.switch_samplers(sampler)
        pipeline_type = pipeline_type or self.default_pipeline_type
        seed_all(seed)

        views_dict = {'front': front_image}
        if back_image is not None: views_dict['back'] = back_image
        if left_image is not None: views_dict['left'] = left_image
        if right_image is not None: views_dict['right'] = right_image
        views_list = list(views_dict.keys())

        if preprocess_image:
            views_dict = {k: self.preprocess_image(v) for k, v in views_dict.items()}

        # Conditioning per view
        conds_512, conds_1024 = {}, {}
        for v, img in views_dict.items():
            conds_512[v] = self.get_cond([img], 512)
            if pipeline_type != '512':
                conds_1024[v] = self.get_cond([img], 1024)

        # Sparse structure
        ss_res = {'512': 32, '1024': 64, '1024_cascade': 32, '1536_cascade': 32}[pipeline_type]
        coords = self.sample_sparse_structure_multiview(
            conds_512, views_list, ss_res, num_samples, sparse_structure_sampler_params,
            front_axis, blend_temperature,
        )

        # Shape
        if pipeline_type == '512':
            shape_slat = self.sample_shape_slat_multiview(
                conds_512, views_list, self.models['shape_slat_flow_model_512'],
                coords, shape_slat_sampler_params, front_axis, blend_temperature,
            )
            res = 512
        elif pipeline_type == '1024':
            shape_slat = self.sample_shape_slat_multiview(
                conds_1024, views_list, self.models['shape_slat_flow_model_1024'],
                coords, shape_slat_sampler_params, front_axis, blend_temperature,
            )
            res = 1024
        elif pipeline_type in ('1024_cascade', '1536_cascade'):
            target = 1024 if pipeline_type == '1024_cascade' else 1536
            shape_slat, res = self.sample_shape_slat_cascade(
                conds_512[views_list[0]], conds_1024[views_list[0]],
                self.models['shape_slat_flow_model_512'], self.models['shape_slat_flow_model_1024'],
                512, target, coords, shape_slat_sampler_params, max_num_tokens,
            )

        # Texture
        tex_conds = conds_1024 if pipeline_type != '512' else conds_512
        tex_model_key = 'tex_slat_flow_model_512' if pipeline_type == '512' else 'tex_slat_flow_model_1024'
        tex_slat = self.sample_tex_slat_multiview(
            tex_conds, views_list, shape_slat, self.models[tex_model_key],
            tex_slat_sampler_params, front_axis, blend_temperature,
        )

        torch.cuda.empty_cache()
        out_mesh = self.decode_latent(shape_slat, tex_slat, res)
        torch.cuda.empty_cache()
        return out_mesh
```

- [ ] **Step 7: Commit**

```bash
git add trellis2/pipelines/trellis2_image_to_3d.py
git commit -m "feat: add multi-view generation methods to image-to-3D pipeline"
```

### Task 3: Add multi-view texturing to texturing pipeline

**Files:**
- Modify: `trellis2/pipelines/trellis2_texturing.py`

- [ ] **Step 1: Add `run_multiview` method to `Trellis2TexturingPipeline`**

Add after the existing `run` method:

```python
    @torch.no_grad()
    def run_multiview(
        self,
        mesh: 'trimesh.Trimesh',
        front_image: Image.Image,
        back_image: Optional[Image.Image] = None,
        left_image: Optional[Image.Image] = None,
        right_image: Optional[Image.Image] = None,
        seed: int = 42,
        tex_slat_sampler_params: dict = {},
        preprocess_image: bool = True,
        resolution: int = 1024,
        texture_size: int = 2048,
        front_axis: str = 'z',
        blend_temperature: float = 2.0,
    ) -> 'trimesh.Trimesh':
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        views_dict = {'front': front_image}
        if back_image is not None: views_dict['back'] = back_image
        if left_image is not None: views_dict['left'] = left_image
        if right_image is not None: views_dict['right'] = right_image
        views_list = list(views_dict.keys())

        if preprocess_image:
            views_dict = {k: self.preprocess_image(v) for k, v in views_dict.items()}

        mesh = self.preprocess_mesh(mesh)
        cond_res = 512 if resolution == 512 else 1024

        conds = {}
        for v, img in views_dict.items():
            conds[v] = self.get_cond([img], cond_res)

        shape_slat = self.encode_shape_slat(mesh, resolution)

        # Multi-view tex slat sampling
        std = torch.tensor(self.shape_slat_normalization['std'])[None].to(shape_slat.device)
        mean = torch.tensor(self.shape_slat_normalization['mean'])[None].to(shape_slat.device)
        shape_slat_normalized = (shape_slat - mean) / std

        flow_model = self.models[f'tex_slat_flow_model_{cond_res}']
        in_channels = flow_model.in_channels
        noise = shape_slat.replace(
            feats=torch.randn(shape_slat.coords.shape[0], in_channels - shape_slat.feats.shape[1]).to(self.device)
        )

        from .samplers import FlowEulerMultiViewGuidanceIntervalSampler
        sampler = FlowEulerMultiViewGuidanceIntervalSampler(
            sigma_min=self.tex_slat_sampler.sigma_min,
            resolution=flow_model.resolution,
        )
        sampler_params = {**self.tex_slat_sampler_params, **tex_slat_sampler_params}

        slat = sampler.sample(
            flow_model, noise, conds=conds, views=views_list,
            front_axis=front_axis, blend_temperature=blend_temperature,
            concat_cond=shape_slat_normalized,
            **sampler_params, verbose=True, tqdm_desc="Sampling texture SLat (MultiView)",
        ).samples

        std_t = torch.tensor(self.tex_slat_normalization['std'])[None].to(slat.device)
        mean_t = torch.tensor(self.tex_slat_normalization['mean'])[None].to(slat.device)
        slat = slat * std_t + mean_t

        pbr_voxel = self.decode_tex_slat(slat)
        torch.cuda.empty_cache()
        return self.postprocess_mesh(mesh, pbr_voxel, resolution, texture_size)
```

- [ ] **Step 2: Commit**

```bash
git add trellis2/pipelines/trellis2_texturing.py
git commit -m "feat: add multi-view texturing method"
```

---

## Chunk 3: Mesh Processing Utilities

### Task 4: Create mesh processing utility module

**Files:**
- Create: `trellis2/utils/mesh_processing.py`

- [ ] **Step 1: Create the mesh processing module**

Create `trellis2/utils/mesh_processing.py`:

```python
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
        v_out, f_out = cm.get_vertices(), cm.get_faces()
    else:
        import mrmeshpy
        mr_mesh = mrmeshpy.Mesh()
        mr_mesh.points = mrmeshpy.PointCloud()
        verts_np = mesh.vertices.astype(np.float32)
        faces_np = mesh.faces.astype(np.int32)
        mr_mesh = mrmeshpy.meshFromFV(
            [mrmeshpy.Vector3f(*v) for v in verts_np],
            [mrmeshpy.ThreeVertIds(*f) for f in faces_np],
        )
        settings = mrmeshpy.DecimateSettings()
        settings.maxDeletedFaces = max(0, len(mesh.faces) - target_face_num)
        settings.packMesh = True
        mrmeshpy.decimateMesh(mr_mesh, settings)
        v_out = torch.from_numpy(np.array([[p.x, p.y, p.z] for p in mr_mesh.points.vec])).float()
        f_out = torch.from_numpy(np.array([[f[0], f[1], f[2]] for f in mr_mesh.topology.getValidFaces()])).int()

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
        import mrmeshpy
        verts = mesh.vertices.astype(np.float32)
        faces = mesh.faces.astype(np.int32)
        mr_mesh = mrmeshpy.meshFromFV(
            [mrmeshpy.Vector3f(*v) for v in verts],
            [mrmeshpy.ThreeVertIds(*f) for f in faces],
        )
        hole_edges = mr_mesh.topology.findHoleRepresentiveEdges()
        for edge in hole_edges:
            params = mrmeshpy.FillHoleParams()
            params.metric = mrmeshpy.getUniversalMetric(mr_mesh)
            mrmeshpy.fillHole(mr_mesh, edge, params)
        v_out = np.array([[p.x, p.y, p.z] for p in mr_mesh.points.vec], dtype=np.float32)
        f_valid = mr_mesh.topology.getValidFaces()
        f_out = np.array([[f[0], f[1], f[2]] for f in f_valid], dtype=np.int32)
        return trimesh.Trimesh(vertices=v_out, faces=f_out, process=False)
    else:
        import cumesh
        vertices = torch.from_numpy(mesh.vertices).float().cuda()
        faces = torch.from_numpy(mesh.faces).int().cuda()
        cm = cumesh.CuMesh()
        cm.init(vertices, faces)
        cm.fill_holes(max_hole_perimeter=max_perimeter)
        return trimesh.Trimesh(
            vertices=cm.get_vertices().cpu().numpy(),
            faces=cm.get_faces().cpu().numpy(),
            process=False,
        )


def smooth_normals(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Compute smooth vertex normals."""
    new_mesh = mesh.copy()
    new_mesh.vertex_normals = trimesh.smoothing.get_vertices_normals(mesh)
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
        o3d_mesh = o3d_mesh.filter_smooth_laplacian(iterations)
    else:
        o3d_mesh = o3d_mesh.filter_smooth_taubin(iterations)

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
```

- [ ] **Step 2: Commit**

```bash
git add trellis2/utils/mesh_processing.py
git commit -m "feat: add mesh processing utilities (remesh, simplify, fill holes, smooth)"
```

---

## Chunk 4: FastAPI Server

### Task 5: Create FastAPI server

**Files:**
- Create: `trellis2_api.py`
- Create: `requirements-api.txt`

- [ ] **Step 1: Create `requirements-api.txt`**

```
# Web framework
fastapi==0.115.5
uvicorn[standard]==0.32.1
python-multipart==0.0.17

# Image processing
pillow>=10.0.0
opencv-python-headless>=4.8.0
imageio>=2.30.0
imageio-ffmpeg>=0.5.0

# ML / vision
timm>=0.9.0
transformers>=4.40.0
huggingface_hub>=0.20.0
einops>=0.7.0

# 3D geometry
trimesh>=4.0.0
open3d>=0.18.0

# Utilities
tqdm>=4.65.0
easydict>=1.10
pydantic>=2.0.0
numpy
```

- [ ] **Step 2: Create `trellis2_api.py`**

```python
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
        tex_pipeline = Trellis2TexturingPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
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
                mesh.export(glb_path)
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
                mesh.export(glb_path)
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
    import trimesh as _trimesh

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
        mesh = _trimesh.load(mesh_path)

        async with gpu_lock:
            await run_in_threadpool(ensure_model_loaded)
            logger.info(f"[{request_id}] Texturing mesh...")

            def _run():
                result = tex_pipeline.run(mesh, image, seed=seed, resolution=resolution, texture_size=texture_size)
                out_path = os.path.join(req_dir, "output.glb")
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
    import trimesh as _trimesh

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
        mesh = _trimesh.load(mesh_path)

        async with gpu_lock:
            await run_in_threadpool(ensure_model_loaded)
            logger.info(f"[{request_id}] Texturing mesh (multi-view)...")

            def _run():
                result = tex_pipeline.run_multiview(
                    mesh, front_image=front_img, back_image=back_img,
                    left_image=left_img, right_image=right_img,
                    seed=seed, resolution=resolution, texture_size=texture_size,
                    front_axis=front_axis, blend_temperature=blend_temperature,
                )
                out_path = os.path.join(req_dir, "output.glb")
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


@app.post("/mesh-process")
async def mesh_process(
    mesh_file: UploadFile = File(...),
    operation: str = Form(..., description="Operation: simplify, remesh, fill_holes, smooth_normals, laplacian_smooth, weld_vertices"),
    target_face_num: int = Form(100000),
    resolution: int = Form(512),
    iterations: int = Form(5),
    method: str = Form("cumesh"),
):
    """Apply mesh processing operations."""
    import trimesh as _trimesh
    from trellis2.utils.mesh_processing import (
        simplify_mesh, remesh_dual_contouring, fill_holes,
        smooth_normals, laplacian_smooth, weld_vertices,
    )

    try:
        request_id = str(uuid.uuid4())
        req_dir = os.path.join(OUTPUT_DIR, request_id)
        os.makedirs(req_dir, exist_ok=True)

        mesh_path = os.path.join(req_dir, f"input{os.path.splitext(mesh_file.filename)[1]}")
        with open(mesh_path, "wb") as buf:
            shutil.copyfileobj(mesh_file.file, buf)
        mesh = _trimesh.load(mesh_path)

        def _run():
            if operation == "simplify":
                return simplify_mesh(mesh, target_face_num, method=method)
            elif operation == "remesh":
                return remesh_dual_contouring(mesh, resolution=resolution)
            elif operation == "fill_holes":
                return fill_holes(mesh, method=method)
            elif operation == "smooth_normals":
                return smooth_normals(mesh)
            elif operation == "laplacian_smooth":
                return laplacian_smooth(mesh, iterations=iterations, method=method)
            elif operation == "weld_vertices":
                return weld_vertices(mesh)
            else:
                raise ValueError(f"Unknown operation: {operation}")

        result = await run_in_threadpool(_run)

        out_path = os.path.join(req_dir, "output.glb")
        result.export(out_path)

        return {
            "request_id": request_id,
            "glb_url": f"/download/{request_id}/output.glb",
            "vertices": len(result.vertices),
            "faces": len(result.faces),
        }
    except Exception as e:
        logger.error(f"Mesh-process failed: {e}", exc_info=True)
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
```

- [ ] **Step 3: Commit**

```bash
git add trellis2_api.py requirements-api.txt
git commit -m "feat: add FastAPI server with generate, multiview, texture, and mesh-process endpoints"
```

---

## Chunk 5: Docker Packaging

### Task 6: Create Docker configuration

**Files:**
- Create: `Dockerfile`
- Create: `docker-compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Create Dockerfile**

Following ReconViaGen pattern:

```dockerfile
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
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
version: "3.8"

# =============================================================================
# TRELLIS.2 API — Docker Compose
#
# Quick start:
#   cp .env.example .env && $EDITOR .env
#   docker compose up -d --build
# =============================================================================

services:
  trellis2:
    build:
      context: .
      dockerfile: Dockerfile
      args:
        TORCH_CUDA_ARCH_LIST: "12.0"
        MAX_JOBS: "4"

    image: trellis2-api:latest
    container_name: trellis2-api

    ports:
      - "${API_PORT:-52070}:52070"

    volumes:
      - ${HF_CACHE_PATH}:/app/hf_cache:rw
      - ./logs:/app/logs

    environment:
      - SPCONV_ALGO=native
      - HF_HOME=/app/hf_cache
      - TRANSFORMERS_CACHE=/app/hf_cache
      - HUGGINGFACE_HUB_CACHE=/app/hf_cache

    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ["${GPU_ID}"]
              capabilities: [gpu]

    shm_size: "8gb"
    restart: unless-stopped
```

- [ ] **Step 3: Create .env.example**

```
# =============================================================================
# TRELLIS.2 API — Machine-specific configuration
#
# cp .env.example .env && $EDITOR .env
# =============================================================================

# Host path for HuggingFace model cache
HF_CACHE_PATH=/path/to/huggingface/cache

# GPU device index (0-based)
GPU_ID=0

# API port on host
API_PORT=52070

# (Optional) HuggingFace token for gated models
# HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
```

- [ ] **Step 4: Add .env to .gitignore**

Append `.env` to `.gitignore` if not already present.

- [ ] **Step 5: Commit**

```bash
git add Dockerfile docker-compose.yml .env.example .gitignore
git commit -m "feat: add Docker packaging for TRELLIS.2 API service"
```

---

## Chunk 6: Integration Verification

### Task 7: Final verification

- [ ] **Step 1: Verify all imports**

```bash
cd /Users/between2058/Documents/code/TRELLIS.2
python -c "
from trellis2.pipelines.samplers import (
    FlowEulerMultiViewSampler, FlowEulerMultiViewGuidanceIntervalSampler,
    FlowRK4Sampler, FlowRK4GuidanceIntervalSampler,
    FlowRK4MultiViewSampler, FlowRK4MultiViewGuidanceIntervalSampler,
    FlowHeunSampler, FlowHeunGuidanceIntervalSampler,
    FlowHeunMultiViewSampler, FlowHeunMultiViewGuidanceIntervalSampler,
)
from trellis2.utils.mesh_processing import (
    simplify_mesh, remesh_dual_contouring, fill_holes,
    smooth_normals, laplacian_smooth, weld_vertices,
)
print('All imports OK')
"
```

- [ ] **Step 2: Verify API module imports**

```bash
python -c "import trellis2_api; print('API module OK')"
```

- [ ] **Step 3: Final commit with all changes**

```bash
git add -A
git status
git commit -m "feat: TRELLIS.2 advanced features - multi-view, mesh processing, FastAPI, Docker"
```
