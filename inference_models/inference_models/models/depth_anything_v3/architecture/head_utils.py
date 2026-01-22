# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class Permute(nn.Module):
    """nn.Module wrapper around Tensor.permute for cleaner nn.Sequential usage."""

    dims: Tuple[int, ...]

    def __init__(self, dims: Tuple[int, ...]) -> None:
        super().__init__()
        self.dims = dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.permute(*self.dims)


def position_grid_to_embed(
    pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100
) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)

    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)

    emb = torch.cat([emb_x, emb_y], dim=-1)

    return emb.view(H, W, embed_dim)


def make_sincos_pos_embed(
    embed_dim: int, pos: torch.Tensor, omega_0: float = 100
) -> torch.Tensor:
    """Generate 1D positional embedding from a given grid using sine and cosine functions."""
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=torch.float32, device=pos.device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega

    pos = pos.reshape(-1)
    out = torch.einsum("m,d->md", pos, omega)

    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)

    emb = torch.cat([emb_sin, emb_cos], dim=1)
    return emb.float()


def create_uv_grid(
    width: int,
    height: int,
    aspect_ratio: float = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor:
    """Create a normalized UV grid of shape (width, height, 2)."""
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid


def custom_interpolate(
    x: torch.Tensor,
    size: Union[Tuple[int, int], None] = None,
    scale_factor: Union[float, None] = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """Safe interpolation implementation to avoid INT_MAX overflow."""
    if size is None:
        assert scale_factor is not None, "Either size or scale_factor must be provided."
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))

    INT_MAX = 1610612736
    total = size[0] * size[1] * x.shape[0] * x.shape[1]

    if total > INT_MAX:
        chunks = torch.chunk(x, chunks=(total // INT_MAX) + 1, dim=0)
        outs = [
            F.interpolate(c, size=size, mode=mode, align_corners=align_corners)
            for c in chunks
        ]
        return torch.cat(outs, dim=0).contiguous()

    return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)
