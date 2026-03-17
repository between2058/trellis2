from typing import *
import torch
import numpy as np
from tqdm import tqdm
from easydict import EasyDict as edict
from .base import Sampler
from .classifier_free_guidance_mixin import ClassifierFreeGuidanceSamplerMixin
from .guidance_interval_mixin import GuidanceIntervalSamplerMixin


class FlowEulerSampler(Sampler):
    """
    Generate samples from a flow-matching model using Euler sampling.

    Args:
        sigma_min: The minimum scale of noise in flow.
    """
    def __init__(
        self,
        sigma_min: float,
    ):
        self.sigma_min = sigma_min

    def _eps_to_xstart(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (x_t - (self.sigma_min + (1 - self.sigma_min) * t) * eps) / (1 - t)

    def _xstart_to_eps(self, x_t, t, x_0):
        assert x_t.shape == x_0.shape
        return (x_t - (1 - t) * x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _v_to_xstart_eps(self, x_t, t, v):
        assert x_t.shape == v.shape
        eps = (1 - t) * v + x_t
        x_0 = (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * v
        return x_0, eps
    
    def _pred_to_xstart(self, x_t, t, pred):
        return (1 - self.sigma_min) * x_t - (self.sigma_min + (1 - self.sigma_min) * t) * pred

    def _xstart_to_pred(self, x_t, t, x_0):
        return ((1 - self.sigma_min) * x_t - x_0) / (self.sigma_min + (1 - self.sigma_min) * t)

    def _inference_model(self, model, x_t, t, cond=None, **kwargs):
        t = torch.tensor([1000 * t] * x_t.shape[0], device=x_t.device, dtype=torch.float32)
        return model(x_t, t, cond, **kwargs)

    def _get_model_prediction(self, model, x_t, t, cond=None, **kwargs):
        pred_v = self._inference_model(model, x_t, t, cond, **kwargs)
        pred_x_0, pred_eps = self._v_to_xstart_eps(x_t=x_t, t=t, v=pred_v)
        return pred_x_0, pred_eps, pred_v

    @torch.no_grad()
    def sample_once(
        self,
        model,
        x_t,
        t: float,
        t_prev: float,
        cond: Optional[Any] = None,
        **kwargs
    ):
        """
        Sample x_{t-1} from the model using Euler method.
        
        Args:
            model: The model to sample from.
            x_t: The [N x C x ...] tensor of noisy inputs at time t.
            t: The current timestep.
            t_prev: The previous timestep.
            cond: conditional information.
            **kwargs: Additional arguments for model inference.

        Returns:
            a dict containing the following
            - 'pred_x_prev': x_{t-1}.
            - 'pred_x_0': a prediction of x_0.
        """
        pred_x_0, pred_eps, pred_v = self._get_model_prediction(model, x_t, t, cond, **kwargs)
        pred_x_prev = x_t - (t - t_prev) * pred_v
        return edict({"pred_x_prev": pred_x_prev, "pred_x_0": pred_x_0})

    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond: Optional[Any] = None,
        steps: int = 50,
        rescale_t: float = 1.0,
        verbose: bool = True,
        tqdm_desc: str = "Sampling",
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            verbose: If True, show a progress bar.
            tqdm_desc: A customized tqdm desc.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        sample = noise
        t_seq = np.linspace(1, 0, steps + 1)
        t_seq = rescale_t * t_seq / (1 + (rescale_t - 1) * t_seq)
        t_seq = t_seq.tolist()
        t_pairs = list((t_seq[i], t_seq[i + 1]) for i in range(steps))
        ret = edict({"samples": None, "pred_x_t": [], "pred_x_0": []})
        for t, t_prev in tqdm(t_pairs, desc=tqdm_desc, disable=not verbose):
            out = self.sample_once(model, sample, t, t_prev, cond, **kwargs)
            sample = out.pred_x_prev
            ret.pred_x_t.append(out.pred_x_prev)
            ret.pred_x_0.append(out.pred_x_0)
        ret.samples = sample
        return ret


class FlowEulerCfgSampler(ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, **kwargs)


class FlowEulerGuidanceIntervalSampler(GuidanceIntervalSamplerMixin, ClassifierFreeGuidanceSamplerMixin, FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with classifier-free guidance and interval.
    """
    @torch.no_grad()
    def sample(
        self,
        model,
        noise,
        cond,
        neg_cond,
        steps: int = 50,
        rescale_t: float = 1.0,
        guidance_strength: float = 3.0,
        guidance_interval: Tuple[float, float] = (0.0, 1.0),
        verbose: bool = True,
        **kwargs
    ):
        """
        Generate samples from the model using Euler method.
        
        Args:
            model: The model to sample from.
            noise: The initial noise tensor.
            cond: conditional information.
            neg_cond: negative conditional information.
            steps: The number of steps to sample.
            rescale_t: The rescale factor for t.
            guidance_strength: The strength of classifier-free guidance.
            guidance_interval: The interval for classifier-free guidance.
            verbose: If True, show a progress bar.
            **kwargs: Additional arguments for model_inference.

        Returns:
            a dict containing the following
            - 'samples': the model samples.
            - 'pred_x_t': a list of prediction of x_t.
            - 'pred_x_0': a list of prediction of x_0.
        """
        return super().sample(model, noise, cond, steps, rescale_t, verbose, neg_cond=neg_cond, guidance_strength=guidance_strength, guidance_interval=guidance_interval, **kwargs)


# =============================================================================
# Multi-View Samplers
# =============================================================================

class FlowEulerMultiViewSampler(FlowEulerSampler):
    """
    Generate samples from a flow-matching model using Euler sampling with multi-view blending.
    Each voxel is weighted by its spatial proximity to each view direction.
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


# =============================================================================
# RK4 Samplers
# =============================================================================

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
    """Multi-view RK4 with CFG and guidance interval."""
    pass


# =============================================================================
# Heun (RK2) Samplers
# =============================================================================

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
    """Multi-view Heun with CFG and guidance interval."""
    pass
