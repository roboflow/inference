# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

# Implementation of 2D Rotary Position Embeddings (RoPE).

# Inspired by:
#         https://github.com/meta-llama/codellama/blob/main/llama/model.py
#         https://github.com/naver-ai/rope-vit


from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionGetter:
    """Generates and caches 2D spatial positions for patches in a grid."""

    def __init__(self):
        self.position_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def __call__(
        self, batch_size: int, height: int, width: int, device: torch.device
    ) -> torch.Tensor:
        if (height, width) not in self.position_cache:
            y_coords = torch.arange(height, device=device)
            x_coords = torch.arange(width, device=device)
            positions = torch.cartesian_prod(y_coords, x_coords)
            self.position_cache[height, width] = positions

        cached_positions = self.position_cache[height, width]
        return cached_positions.view(1, height * width, 2).expand(batch_size, -1, -1).clone()


class RotaryPositionEmbedding2D(nn.Module):
    """2D Rotary Position Embedding implementation."""

    def __init__(self, frequency: float = 100.0, scaling_factor: float = 1.0):
        super().__init__()
        self.base_frequency = frequency
        self.scaling_factor = scaling_factor
        self.frequency_cache: Dict[Tuple, Tuple[torch.Tensor, torch.Tensor]] = {}

    def _compute_frequency_components(
        self, dim: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_key = (dim, seq_len, device, dtype)
        if cache_key not in self.frequency_cache:
            exponents = torch.arange(0, dim, 2, device=device).float() / dim
            inv_freq = 1.0 / (self.base_frequency**exponents)

            positions = torch.arange(seq_len, device=device, dtype=inv_freq.dtype)
            angles = torch.einsum("i,j->ij", positions, inv_freq)

            angles = angles.to(dtype)
            angles = torch.cat((angles, angles), dim=-1)
            cos_components = angles.cos().to(dtype)
            sin_components = angles.sin().to(dtype)
            self.frequency_cache[cache_key] = (cos_components, sin_components)

        return self.frequency_cache[cache_key]

    @staticmethod
    def _rotate_features(x: torch.Tensor) -> torch.Tensor:
        feature_dim = x.shape[-1]
        x1, x2 = x[..., : feature_dim // 2], x[..., feature_dim // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _apply_1d_rope(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor,
        cos_comp: torch.Tensor,
        sin_comp: torch.Tensor,
    ) -> torch.Tensor:
        cos = F.embedding(positions, cos_comp)[:, None, :, :]
        sin = F.embedding(positions, sin_comp)[:, None, :, :]
        return (tokens * cos) + (self._rotate_features(tokens) * sin)

    def forward(self, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        assert tokens.size(-1) % 2 == 0, "Feature dimension must be even"
        assert (
            positions.ndim == 3 and positions.shape[-1] == 2
        ), "Positions must have shape (batch_size, n_tokens, 2)"

        feature_dim = tokens.size(-1) // 2

        max_position = int(positions.max()) + 1
        cos_comp, sin_comp = self._compute_frequency_components(
            feature_dim, max_position, tokens.device, tokens.dtype
        )

        vertical_features, horizontal_features = tokens.chunk(2, dim=-1)

        vertical_features = self._apply_1d_rope(
            vertical_features, positions[..., 0], cos_comp, sin_comp
        )
        horizontal_features = self._apply_1d_rope(
            horizontal_features, positions[..., 1], cos_comp, sin_comp
        )

        return torch.cat((vertical_features, horizontal_features), dim=-1)

