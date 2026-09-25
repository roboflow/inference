# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# Adapted from Meta V-JEPA 2.1 for video inference only.
# See UPSTREAM.md for the source revision and LICENSE for the MIT license.

import torch
from torch import nn
from torch.nn import functional as F


class PatchEmbed3D(nn.Module):
    def __init__(self, tubelet_size):
        super().__init__()
        self.proj = nn.Conv3d(
            3, 768, kernel_size=(tubelet_size, 16, 16), stride=(tubelet_size, 16, 16)
        )

    def forward(self, video):
        return self.proj(video).flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, tokens):
        return self.fc2(F.gelu(self.fc1(tokens)))


def rotate_pairs(values, positions):
    width = values.shape[-1]
    # Preserve Meta's frequency dtype, including rounding under autocast.
    frequencies = torch.arange(width // 2, dtype=values.dtype, device=values.device)
    frequencies /= width / 2.0
    frequencies = 1.0 / 10000**frequencies
    angles = torch.einsum("..., f -> ... f", positions, frequencies)
    sine = angles.sin().repeat_interleave(2, dim=-1)
    cosine = angles.cos().repeat_interleave(2, dim=-1)
    first, second = values.unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((-second, first), dim=-1).flatten(-2)
    return values * cosine + rotated * sine


class RoPEAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(768, 3 * 768)
        self.proj = nn.Linear(768, 768)

    def forward(self, tokens, positions):
        batch, count, dim = tokens.shape
        queries, keys, values = (
            self.qkv(tokens).unflatten(-1, (3, 12, 64)).permute(2, 0, 3, 1, 4)
        ).unbind(0)
        # Each axis rotates 20 channels. The final four channels stay unrotated.
        queries = torch.cat(
            [
                rotate_pairs(queries[..., start : start + 20], position)
                for start, position in zip((0, 20, 40), positions)
            ]
            + [queries[..., 60:]],
            dim=-1,
        )
        keys = torch.cat(
            [
                rotate_pairs(keys[..., start : start + 20], position)
                for start, position in zip((0, 20, 40), positions)
            ]
            + [keys[..., 60:]],
            dim=-1,
        )
        # RoPE promotes Q/K while V can remain FP16; SDPA needs matching dtypes.
        # Keep the established BF16 autocast path unchanged.
        if values.dtype == torch.float16:
            queries, keys = queries.to(values.dtype), keys.to(values.dtype)
        with torch.backends.cuda.sdp_kernel():
            attended = F.scaled_dot_product_attention(queries, keys, values)
        return self.proj(attended.transpose(1, 2).reshape(batch, count, dim))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(768, eps=1e-6)
        self.attn = RoPEAttention()
        self.norm2 = nn.LayerNorm(768, eps=1e-6)
        self.mlp = MLP(768)

    def forward(self, tokens, positions):
        tokens = tokens + self.attn(self.norm1(tokens), positions)
        return tokens + self.mlp(self.norm2(tokens))


class VJepaEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = PatchEmbed3D(tubelet_size=2)
        self.blocks = nn.ModuleList([Block() for _ in range(12)])
        self.video_mod_embed = nn.Parameter(torch.empty(1, 1, 768))
        # Full trainer checkpoints include these image and intermediate-output
        # parameters. Retain their keys instead of weakening strict loading.
        self.patch_embed_img = PatchEmbed3D(tubelet_size=1)
        self.img_mod_embed = nn.Parameter(torch.empty(1, 1, 768))
        self.norms_block = nn.ModuleList(
            [nn.LayerNorm(768, eps=1e-6) for _ in range(4)]
        )

    def forward(self, video):
        _, _, frames, height, width = video.shape
        depth, rows, columns = frames // 2, height // 16, width // 16
        indices = torch.arange(depth * rows * columns, device=video.device)
        temporal = 1.0 * (indices // (rows * columns))
        vertical = 1.0 * ((indices % (rows * columns)) // columns)
        horizontal = 1.0 * (indices % columns)
        # Meta interpolates spatial coordinates onto its 16x16 pretraining grid;
        # temporal positions remain in tubelet units.
        positions = (
            temporal,
            vertical * 15 / (rows - 1),
            horizontal * 15 / (columns - 1),
        )
        tokens = self.patch_embed(video)
        tokens += self.video_mod_embed.repeat(tokens.shape[0], 1, 1)
        for block in self.blocks:
            tokens = block(tokens, positions)
        return self.norms_block[-1](tokens)


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, 2 * dim)

    def forward(self, queries, tokens):
        batch, count, dim = queries.shape
        queries = (
            self.q(queries)
            .reshape(batch, count, self.num_heads, dim // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        keys, values = (
            self.kv(tokens)
            .reshape(batch, tokens.shape[1], 2, self.num_heads, dim // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        ).unbind(0)
        with torch.backends.cuda.sdp_kernel():
            attended = F.scaled_dot_product_attention(queries, keys, values)
        return attended.transpose(1, 2).reshape(batch, count, dim)


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.xattn = CrossAttention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim)

    def forward(self, queries, tokens):
        # Meta normalizes the memory, not the queries, before cross-attention
        # and has no attention output projection.
        queries = queries + self.xattn(queries, self.norm1(tokens))
        return queries + self.mlp(self.norm2(queries))
