from typing import Optional
import numpy as np
import torch
from timm.models.vision_transformer import DropPath, Mlp

from lidra.model.backbone.mcc.common import (
    Transformer,
    PatchEmbed,
    PositionalEncoding2D,
)


class DensityDecoder(torch.nn.Module):
    def __init__(
        self,
        transformer: Transformer,
        image_size=224,
        image_channels=3,
        image_patch_size=16,
        input_n_cls_tokens=0,
        max_n_queries=256000,
    ):
        super().__init__()

        # TODO(Pierre) : clean / remove temporary patch_embed
        patch_embed = PatchEmbed(
            image_size,
            image_patch_size,
            image_channels,
            transformer.embed_dim,
        )
        self.latent_pe = PositionalEncoding2D(
            transformer.embed_dim,
            grid_size=int(patch_embed.num_patches**0.5),
            n_cls_token=input_n_cls_tokens,
        )
        self.xyz_pe = PositionalEncodingXYZ(transformer.embed_dim, scale=1.0)
        self.transformer = transformer

        self.max_n_queries = max_n_queries

    def _get_chunk_i(self, xyz, i):
        i_start = i * self.max_n_queries
        i_end = (i + 1) * self.max_n_queries
        return xyz[:, i_start:i_end]

    def _forward_chunk(self, latent, xyz_chunk):
        # prepare positions
        xyz = self.xyz_pe(xyz_chunk)

        # apply transorfmer
        x = torch.cat([latent, xyz], dim=1)
        x = self.transformer(x, unseen_size=xyz.shape[1])

        return x

    def forward(self, latent, xyz):
        # prepare latent
        latent = self.latent_pe(latent)

        # Chunked prediction if >max_n_queries, to prevent OOM
        xs = []
        n_queries = xyz.shape[1]
        n_chunks = int(np.ceil(n_queries / self.max_n_queries))
        for i in range(n_chunks):
            xyz_i = self._get_chunk_i(xyz, i)
            xs.append(self._forward_chunk(latent, xyz_i))

        return torch.cat(xs, dim=1)


class PositionalEncodingXYZ(torch.nn.Module):
    def __init__(
        self,
        num_pos_feats: int = 64,
        n_dims: int = 3,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()

        if scale is None or scale <= 0.0:
            scale = 1.0

        assert num_pos_feats % 2 == 0
        num_pos_feats = num_pos_feats // 2
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((n_dims, num_pos_feats)),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = coords
        x = x @ self.positional_encoding_gaussian_matrix
        x = 2 * torch.pi * x
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x


class LayerScale(torch.nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = torch.nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class DecoderAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"

        self.enc_self_attn = torch.nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attn_drop,
            batch_first=True,
            bias=qkv_bias,
        )
        self.dec_x_attn = torch.nn.MultiheadAttention(
            dim,
            num_heads,
            dropout=attn_drop,
            batch_first=True,
            bias=qkv_bias,
        )

    def forward(self, x, unseen_size):
        # split into encoder (B, N_e, D) and decoder (B, N_d, D) tokens
        enc, dec = x[:, :-unseen_size], x[:, -unseen_size:]

        # self-attention on encoder tokens (B, N_e, D)
        enc, _ = self.enc_self_attn(enc, enc, enc)

        # cross-attention with query tokens (B, N_q, D)
        dec, _ = self.dec_x_attn(dec, enc, enc)

        # concatenate encoder and decoder tokens back together
        return torch.cat([enc, dec], dim=1)


class DecoderBlock(torch.nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=torch.nn.GELU,
        norm_layer=torch.nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = DecoderAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else torch.nn.Identity()
        )
        self.drop_path1 = (
            DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        )

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values)
            if init_values
            else torch.nn.Identity()
        )
        self.drop_path2 = (
            DropPath(drop_path) if drop_path > 0.0 else torch.nn.Identity()
        )

    def forward(self, x, unseen_size):
        rx = self.norm1(x)
        rx = self.attn(rx, unseen_size)
        rx = self.ls1(rx)
        rx = self.drop_path1(rx)
        x = x + rx

        rx = self.norm2(x)
        rx = self.mlp(rx)
        rx = self.ls2(rx)
        rx = self.drop_path2(rx)
        x = x + rx

        return x
