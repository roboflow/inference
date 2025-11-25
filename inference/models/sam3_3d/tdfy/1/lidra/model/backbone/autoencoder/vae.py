from typing import Optional, Tuple
import numpy as np

import torch
from torch import nn

from lidra.model.layers.drop_path import DropPath
from timm.models.vision_transformer import Block, Mlp, DropPath
from lidra.model.backbone.mcc.common import Transformer
from lidra.model.backbone.mcc.decoder import LayerScale


def create_autoencoder(
    dim: int = 512,
    M: int = 512,
    latent_dim: int = 64,
    N: int = 2048,
    depth: int = 24,
    deterministic: bool = False,
    num_heads: int = 8,
    **kwargs,
):
    if deterministic:
        raise NotImplementedError("Deterministic Autoencoder not implemented")
    else:
        model = KLAutoEncoder(
            n_blocks=depth,
            embed_dim=dim,
            queries_dim=dim,
            output_dim=1,
            num_latents=M,
            latent_dim=latent_dim,
            num_heads=num_heads,
            # dim_head = 64,
            **kwargs,
        )
    return model


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int = 512,
        num_latents: int = 512,
        latent_dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.point_embed = PointEmbed(dim=embed_dim)
        self.learned_latents = nn.Embedding(num_latents, embed_dim)
        self.encoder_cross_attend_blocks = XAttnBlock(
            dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=False,
            drop=0.0,
            attn_drop=0.0,
            init_values=None,
            drop_path=0.0,
            act_layer=torch.nn.GELU,
            norm_layer=torch.nn.LayerNorm,
        )
        self.mean_and_logvar_fc = nn.Linear(embed_dim, latent_dim * 2)

    @staticmethod
    def sample_vae_encoding(mean_and_logvar: torch.Tensor, with_kl: bool):
        mean, logvar = mean_and_logvar.chunk(2, dim=-1)
        posterior = DiagonalGaussianDistribution(mean, logvar)
        x = posterior.sample()
        if with_kl:
            kl = posterior.kl()
            return kl, x
        return x

    def forward(
        self,
        pc: torch.Tensor,
        with_kl: bool = False,
        with_mean_and_logvar=False,
    ):
        B, N, D = pc.shape

        # 3DShape2VecSet uses FPS sampling, but we use learned latents
        latent_embed = self.learned_latents.weight.unsqueeze(0).repeat(B, 1, 1)
        pc_embeddings = self.point_embed(pc)

        x = self.encoder_cross_attend_blocks(latent_embed, context=pc_embeddings)
        mean_and_logvar = self.mean_and_logvar_fc(x)
        if with_mean_and_logvar:
            return (
                self.sample_vae_encoding(mean_and_logvar, with_kl),
                mean_and_logvar,
            )
        return self.sample_vae_encoding(mean_and_logvar, with_kl)


class Decoder(nn.Module):
    def __init__(
        self,
        n_blocks: int = 24,
        embed_dim: int = 512,
        latent_dim: int = 64,
        queries_dim: int = 512,
        output_dim: int = 1,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_path_rate: float = 0.1,
        query_point_embed: Optional[torch.nn.Module] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if query_point_embed is None:
            self.query_point_embed = PointEmbed(dim=embed_dim)
        else:
            self.query_point_embed = query_point_embed

        self.decoder_proj_fc = nn.Linear(latent_dim, embed_dim)
        self.decoder_outputs_fc = (
            nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()
        )

        self.decoder_transformer = Transformer(
            embed_dim=embed_dim,
            n_blocks=n_blocks,
            n_heads=num_heads,
            drop_path=drop_path_rate,
            mlp_ratio=mlp_ratio,
            block_fn=Block,
            n_cls_tokens=0,
        )
        self.decoder_xattn_block = XAttnBlock(
            dim=queries_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

    def forward(self, x: torch.Tensor, queries: torch.Tensor):
        """
        Args:
            x: B x num_latents x D
            queries: B x Q x 3
        Returns:
            B x Q x output_dim
        """
        queries_embeddings = self.query_point_embed(queries)
        x = self.decoder_proj_fc(x)
        x = self.decoder_transformer(x)
        latents = self.decoder_xattn_block(queries_embeddings, context=x)
        return self.decoder_outputs_fc(latents)


class KLAutoEncoder(nn.Module):
    def __init__(
        self,
        *,
        n_blocks: int = 24,
        embed_dim: int = 512,
        queries_dim: int = 512,
        output_dim: int = 1,
        num_latents: int = 512,
        latent_dim: int = 64,
        num_heads: int = 8,
        drop_path_rate: float = 0.1,
        mlp_ratio: float = 4.0,
        share_point_embed_encoder_decoder: bool = False,
    ):
        super().__init__()

        # TODO(Pierre) : kept `queries_dim` for backward compatibility, but we should consider removing it
        assert embed_dim == queries_dim, "embed_dim != queries_dim is not allowed"

        self.encoder = Encoder(
            embed_dim=embed_dim,
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
        )

        self.decoder = Decoder(
            n_blocks=n_blocks,
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            queries_dim=queries_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path_rate,
            query_point_embed=(
                # Whether or not to share seems not that important.
                self.encoder.point_embed
                if share_point_embed_encoder_decoder
                else None
            ),
        )

    def encode(self, *args, **wargs):
        return self.encoder(*args, **wargs)

    def decode(self, *args, **wargs):
        return self.decoder(*args, **wargs)

    def forward(self, pc, queries):
        kl, x = self.encode(pc, with_kl=True)
        o = self.decode(x, queries).squeeze(-1)
        return {"logits": o, "kl": kl}


class DiagonalGaussianDistribution(object):
    """
    Huggingface implementation of DiagonalGaussianDistribution
    https://github.com/huggingface/diffusers/blob/dac623b59f52c58383a39207d5147aa34e0047cd/src/diffusers/models/autoencoders/vae.py#L767
    """

    def __init__(
        self, mean: torch.Tensor, logvar: torch.Tensor, deterministic: bool = False
    ):
        self.mean = mean
        self.logvar = logvar
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.mean.device, dtype=self.mean.dtype
            )

    def sample(self, generator: Optional[torch.Generator] = None) -> torch.Tensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = torch.randn(
            self.mean.shape,
            # generator=generator,
            device=self.mean.device,
            dtype=self.mean.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.mean(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2],
                )
            else:
                return 0.5 * torch.mean(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(
        self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]
    ) -> torch.Tensor:
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class XAttnBlock(torch.nn.Module):
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
        self.attn = Attention(
            dim,
            heads=num_heads,
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

    def forward(self, x, context=None):
        rx = self.norm1(x)
        rx = self.attn(rx, context)
        rx = self.ls1(rx)
        rx = self.drop_path1(rx)
        x = x + rx

        rx = self.norm2(x)
        rx = self.mlp(rx)
        rx = self.ls2(rx)
        rx = self.drop_path2(rx)
        x = x + rx

        return x


class Attention(nn.Module):
    def __init__(
        self, query_dim, context_dim=None, heads=8, dim_head=64, drop_path_rate=0.0
    ):
        super().__init__()
        if context_dim is not None:
            assert (
                query_dim == context_dim
            ), f"query_dim should be equal to context_dim {query_dim} != {context_dim}"
        assert (
            query_dim % heads == 0
        ), f"query_dim {query_dim} must be divisible by heads {heads}"

        context_dim = default(context_dim, query_dim)
        self.embed_dim = query_dim
        self.heads = heads

        # PyTorch MultiheadAttention expects (seq_len, batch, embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=heads, batch_first=True
        )

        self.drop_path = (
            DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
        )

    def forward(self, x, context=None, mask=None):
        context = default(context, x)
        # Apply multihead attention
        out, _ = self.mha(query=x, key=context, value=context, need_weights=False)
        return self.drop_path(out)


class PointEmbed(nn.Module):
    """Embeds 3D points using sinusoidal positional encoding and an MLP."""

    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()
        # Ensure hidden_dim is divisible by 6 since we'll split it into 3 equal parts
        assert hidden_dim % 6 == 0, "hidden_dim must be divisible by 6"

        self.embedding_dim = hidden_dim
        freq_multiplier = hidden_dim // 6

        # Create frequency basis for positional encoding
        frequencies = 2.0 ** torch.arange(freq_multiplier, dtype=torch.float32) * np.pi

        # Create basis vectors for x, y, z coordinates
        basis = torch.zeros(
            3, hidden_dim // 2
        )  # Divide by 2 because we'll use sin and cos
        basis[0, :freq_multiplier] = frequencies  # x-axis frequencies
        basis[1, freq_multiplier : 2 * freq_multiplier] = (
            frequencies  # y-axis frequencies
        )
        basis[2, 2 * freq_multiplier :] = frequencies  # z-axis frequencies

        # Register basis as a buffer (persistent state that's not a parameter)
        self.register_buffer("basis", basis)

        # MLP to process concatenated embeddings and original input
        self.mlp = nn.Linear(hidden_dim + 3, dim)

    def compute_positional_encoding(self, points, basis):
        """
        Compute sinusoidal positional encoding for input points.

        Args:
            points: Input points tensor of shape (batch_size, num_points, 3)
            basis: Frequency basis tensor of shape (3, embedding_dim//2)

        Returns:
            Positional encoding tensor of shape (batch_size, num_points, embedding_dim)
        """
        # Project points onto frequency basis
        projections = torch.matmul(points, basis)  # Equivalent to einsum('bnd,de->bne')

        # Compute sin and cos embeddings and concatenate
        return torch.cat([projections.sin(), projections.cos()], dim=-1)

    def forward(self, points):
        """
        Forward pass to embed 3D points.

        Args:
            points: Input points tensor of shape (batch_size, num_points, 3)

        Returns:
            Embedded points tensor of shape (batch_size, num_points, dim)
        """
        # Compute positional encoding and concatenate with original points
        pos_encoding = self.compute_positional_encoding(points, self.basis)
        features = torch.cat([pos_encoding, points], dim=-1)

        # Transform through MLP
        return self.mlp(features)
