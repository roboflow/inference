from functools import partial
from numbers import Number
import torch
import random
from torch.utils import _pytree
from torch.utils._pytree import tree_map_only
from loguru import logger


def _zeros_like(struct):
    def make_zeros(x):
        if isinstance(x, torch.Tensor):
            return torch.zeros_like(x)
        return x

    return _pytree.tree_map(make_zeros, struct)


def zero_out(args, kwargs):
    args = _zeros_like(args)
    kwargs = _zeros_like(kwargs)
    return args, kwargs


def discard(args, kwargs):
    return (), {}


def _drop_tensors(struct):
    """
    Drop any conditioning that are tensors
    Not using _pytree since we actually want to throw them instead of keeping them.
    """
    if isinstance(struct, dict):
        return {
            k: _drop_tensors(v)
            for k, v in struct.items()
            if not isinstance(v, torch.Tensor)
        }
    elif isinstance(struct, (list, tuple)):
        filtered = [_drop_tensors(x) for x in struct if not isinstance(x, torch.Tensor)]
        return tuple(filtered) if isinstance(struct, tuple) else filtered
    else:
        return struct


def drop_tensors(args, kwargs):
    args = _drop_tensors(args)
    kwargs = _drop_tensors(kwargs)
    return args, kwargs


def add_flag(args, kwargs):
    kwargs["cfg"] = True
    return args, kwargs


class ClassifierFreeGuidance(torch.nn.Module):
    UNCONDITIONAL_HANDLING_TYPES = {
        "zeros": zero_out,
        "discard": discard,
        "drop_tensors": drop_tensors,
        "add_flag": add_flag,
    }

    def __init__(
        self,
        backbone,  # backbone should be a backbone/generator (e.g. DDPM/DDIM/FlowMatching)
        p_unconditional=0.1,
        strength=3.0,
        # "zeros" = set cond tensors to 0,
        # "discard" = remove cond arguments and let underlying model handle it
        # "drop_tensors" = drop all tensors but leave non-tensors
        # "add_flag" = add an argument in kwargs as "cfg" and defer the handling to generator backbone
        unconditional_handling="zeros",
        interval=None,  # only perform cfg if t within interval
    ):
        super().__init__()

        if not (
            unconditional_handling
            in ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES
        ):
            raise RuntimeError(
                f"'{unconditional_handling}' is not valid for `unconditional_handling`, should be in {ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES}"
            )

        self.backbone = backbone
        self.p_unconditional = p_unconditional
        self.strength = strength
        self.unconditional_handling = unconditional_handling
        self.interval = interval
        self._make_unconditional_args = (
            ClassifierFreeGuidance.UNCONDITIONAL_HANDLING_TYPES[
                self.unconditional_handling
            ]
        )

    def _cfg_step_tensor(self, y_cond, y_uncond, strength):
        return (1 + strength) * y_cond - strength * y_uncond

    def _cfg_step(self, y_cond, y_uncond, strength):
        if isinstance(strength, dict):
            return _pytree.tree_map(self._cfg_step_tensor, y_cond, y_uncond, strength)
        else:
            return _pytree.tree_map(
                partial(self._cfg_step_tensor, strength=strength), y_cond, y_uncond
            )

    def inner_forward(self, x, t, is_cond, strength, *args_cond, **kwargs_cond):
        y_cond = self.backbone(x, t, *args_cond, **kwargs_cond)
        if is_cond:
            return y_cond
        else:
            args_cond, kwargs_cond = self._make_unconditional_args(
                args_cond,
                kwargs_cond,
            )
            y_uncond = self.backbone(x, t, *args_cond, **kwargs_cond)
            return self._cfg_step(y_cond, y_uncond, strength)

    def forward(self, x, t, *args_cond, **kwargs_cond):
        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                coin_flip = random.random() < self.p_unconditional
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                strength = get_strength(self.strength, self.interval, t)
                is_cond = not any(x > 0.0 for x in _pytree.tree_flatten(strength)[0])
                return self.inner_forward(
                    x, t, is_cond, strength, *args_cond, **kwargs_cond
                )


def get_strength(strength, interval, t):
    if interval is None:
        return _pytree.tree_map(lambda x: 0.0, strength)

    # If interval is not a dict (single tuple), broadcast it
    if not isinstance(interval, dict):
        return _pytree.tree_map(
            lambda x: x if interval[0] <= t <= interval[1] else 0.0, strength
        )

    return _pytree.tree_map(
        lambda x, iv: x if iv[0] <= t <= iv[1] else 0.0, strength, interval
    )


class PointmapCFG(ClassifierFreeGuidance):

    def __init__(self, *args, strength_pm=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.strength_pm = strength_pm

    def _cfg_step_tensor(self, y_cond, y_uncond, y_unpm, strength, strength_pm):
        # https://arxiv.org/abs/2411.18613
        return y_cond + strength_pm * (y_cond - y_unpm) + strength * (y_unpm - y_uncond)

    def _cfg_step(self, y_cond, y_uncond, y_pm, strength, strength_pm):
        if isinstance(strength, dict):
            return _pytree.tree_map(
                self._cfg_step_tensor, y_cond, y_uncond, y_pm, strength, strength_pm
            )
        else:
            return _pytree.tree_map(
                partial(
                    self._cfg_step_tensor, strength=strength, strength_pm=strength_pm
                ),
                y_cond,
                y_uncond,
                y_pm,
            )

    def inner_forward(
        self, x, t, is_cond, strength, strength_pm, *args_cond, **kwargs_cond
    ):
        y_cond = self.backbone(x, t, *args_cond, **kwargs_cond)

        if is_cond:
            return y_cond
        else:
            force_drop_modalities = (
                self.backbone.condition_embedder.force_drop_modalities
            )
            self.backbone.condition_embedder.force_drop_modalities = [
                "pointmap",
                "rgb_pointmap",
            ]
            y_pm = self.backbone(x, t, *args_cond, **kwargs_cond)
            self.backbone.condition_embedder.force_drop_modalities = (
                force_drop_modalities
            )

            args_cond, kwargs_cond = self._make_unconditional_args(
                args_cond,
                kwargs_cond,
            )
            y_uncond = self.backbone(x, t, *args_cond, **kwargs_cond)
            return self._cfg_step(y_cond, y_uncond, y_pm, strength, strength_pm)

    def forward(self, x, t, *args_cond, **kwargs_cond):
        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                coin_flip = random.random() < self.p_unconditional
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                strength = get_strength(self.strength, self.interval, t)
                is_cond = not any(x > 0.0 for x in _pytree.tree_flatten(strength)[0])
                strength_pm = get_strength(self.strength_pm, self.interval, t)
                return self.inner_forward(
                    x, t, is_cond, strength, strength_pm, *args_cond, **kwargs_cond
                )


class ClassifierFreeGuidanceWithExternalUnconditionalProbability(
    ClassifierFreeGuidance
):

    def __init__(self, *args, use_unconditional_from_flow_matching=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_unconditional_from_flow_matching = use_unconditional_from_flow_matching

    def forward(self, x, t, *args_cond, p_unconditional=None, **kwargs_cond):
        # p_unconditional should be a value in [0, 1], indicating the probability of unconditional

        if p_unconditional is None:
            coin_flip = random.random() < self.p_unconditional
        else:
            coin_flip = random.random() < p_unconditional

        # handle case when no conditional arguments are provided
        if len(args_cond) + len(kwargs_cond) == 0:  # unconditional
            if self.unconditional_handling != "discard":
                raise RuntimeError(
                    f"cannot call `ClassifierFreeGuidance` module without condition"
                )
            return self.backbone(x, t)
        else:  # conditional arguments are provided
            # training mode
            if self.training:
                if coin_flip:  # unconditional
                    args_cond, kwargs_cond = self._make_unconditional_args(
                        args_cond,
                        kwargs_cond,
                    )
                return self.backbone(x, t, *args_cond, **kwargs_cond)
            else:  # inference mode
                strength = get_strength(self.strength, self.interval, t)
                is_cond = not any(x > 0.0 for x in _pytree.tree_flatten(strength)[0])
                return self.inner_forward(
                    x, t, is_cond, strength, *args_cond, **kwargs_cond
                )
