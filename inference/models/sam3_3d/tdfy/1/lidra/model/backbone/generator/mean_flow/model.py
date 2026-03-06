from typing import Callable, Sequence, Union
import torch
import numpy as np
from functools import partial
import optree
import math

from lidra.model.backbone.generator.base import Base
from lidra.data.utils import right_broadcasting
from lidra.data.utils import tree_tensor_map, tree_reduce_unique
from lidra.model.backbone.generator.flow_matching.model import FlowMatching
from torch.nn.attention import SDPBackend, sdpa_kernel


# https://arxiv.org/abs/2505.13447
class MeanFlow(FlowMatching):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def loss(self, x1: torch.Tensor, *args_conditionals, **kwargs_conditionals):
        t = self._generate_t(x1)
        r = self._generate_t(x1)
        # r has to be closer to noise
        t = torch.max(r, t)
        r = torch.min(r, t)
        x0 = self._generate_x0(x1)
        # TODO: Add coupling here for OT flow
        x_t = self._generate_xt(x0, x1, t)
        v = self._generate_target(x0, x1)
        x_t = torch.ones((1, 1000, 8), device=x0.device)
        v = torch.ones((1, 1000, 8), device=x0.device)
        with sdpa_kernel(SDPBackend.MATH):
            u, dudt = torch.func.jvp(
                lambda x_t, t, r: self.reverse_fn(
                    x_t,
                    t,
                    r,
                    *args_conditionals,
                    **kwargs_conditionals,
                ),
                (x_t, t * self.time_scale, r * self.time_scale),
                (
                    v,
                    torch.ones_like(t) * self.time_scale,
                    torch.zeros_like(r) * self.time_scale,
                ),
            )
        u_target = v - (t - r) * dudt.detach()

        # broadcast & and compute loss
        loss = optree.tree_broadcast_map(
            lambda fn, weight, pred, targ: weight * fn(pred, targ),
            self.loss_fn,
            self.loss_weights,
            u,
            u_target,
        )

        loss = sum(optree.tree_flatten(loss)[0])
        return loss
