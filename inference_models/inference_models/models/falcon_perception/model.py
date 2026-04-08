# Copyright 2026 Technology Innovation Institute (TII), Abu Dhabi.
# Licensed under the Apache License, Version 2.0.
# Adapted from https://github.com/tiiuae/Falcon-Perception for integration
# with the inference-models package.
#
# Dense Transformer with hybrid attention, specialized heads (coord/size/seg),
# and AnyUp content-aware upsampler for instance segmentation.

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from inference_models.models.falcon_perception.config import FalconPerceptionConfig


def build_fourier_features(values: torch.Tensor, dim: int) -> torch.Tensor:
    """Encode scalar values as Fourier (sinusoidal) positional features.

    Args:
        values: (B, N) tensor of scalar values in [0, 1].
        dim: Output feature dimension (must be even).

    Returns:
        (B, N, dim) tensor of Fourier features.
    """
    half_dim = dim // 2
    freqs = torch.exp(
        torch.arange(half_dim, device=values.device, dtype=values.dtype)
        * (-math.log(10000.0) / half_dim)
    )
    angles = values.unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0) * math.pi * 2
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


def apply_rotary_embedding(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    """Apply rotary position embedding to input tensor.

    x: (..., head_dim), cos/sin: (..., head_dim).
    Splits head_dim in half, applies rotation, recombines.
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos_half, sin_half = cos[..., :half], sin[..., :half]
    return torch.cat(
        [x1 * cos_half - x2 * sin_half, x2 * cos_half + x1 * sin_half], dim=-1
    )


class GoldenGateRoPE(nn.Module):
    """3D Rotary Position Embedding: 1D sequence + 2D spatial with golden ratio scaling."""

    def __init__(self, config: FalconPerceptionConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.alpha = config.gg_rope_alpha

        # Reference: first half of head_dim gets 1D temporal RoPE,
        # second half gets 2D golden-gate spatial RoPE.
        # We split into 3 parts for simplicity: 1D seq + 2D spatial (h, w)
        self.seq_dim = self.head_dim // 4
        self.spatial_dim = self.head_dim // 4  # per spatial axis
        # Remaining dims get no RoPE (passthrough)
        self.passthrough_dim = self.head_dim - self.seq_dim - 2 * self.spatial_dim

    def forward(
        self,
        seq_len: int,
        image_h_patches: int,
        image_w_patches: int,
        num_image_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute RoPE cos/sin tensors for the full sequence.

        Returns cos, sin each of shape (1, seq_len, head_dim).
        """
        # 1D sequence positions
        positions = torch.arange(seq_len, device=device, dtype=dtype)
        seq_freqs = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.seq_dim, 2, device=device, dtype=dtype)
                / self.seq_dim
            )
        )
        seq_angles = positions.unsqueeze(-1) * seq_freqs.unsqueeze(0)
        seq_cos = torch.cat([seq_angles.cos(), seq_angles.cos()], dim=-1)
        seq_sin = torch.cat([seq_angles.sin(), seq_angles.sin()], dim=-1)

        # 2D spatial positions for image tokens
        spatial_cos_h = torch.ones(seq_len, self.spatial_dim, device=device, dtype=dtype)
        spatial_sin_h = torch.zeros(
            seq_len, self.spatial_dim, device=device, dtype=dtype
        )
        spatial_cos_w = torch.ones(seq_len, self.spatial_dim, device=device, dtype=dtype)
        spatial_sin_w = torch.zeros(
            seq_len, self.spatial_dim, device=device, dtype=dtype
        )

        if num_image_tokens > 0 and image_h_patches > 0 and image_w_patches > 0:
            h_pos = (
                torch.arange(image_h_patches, device=device, dtype=dtype)
                .unsqueeze(1)
                .expand(image_h_patches, image_w_patches)
                .reshape(-1)
            )
            w_pos = (
                torch.arange(image_w_patches, device=device, dtype=dtype)
                .unsqueeze(0)
                .expand(image_h_patches, image_w_patches)
                .reshape(-1)
            )
            n_spatial = min(num_image_tokens, h_pos.shape[0])

            spatial_freqs = 1.0 / (
                (self.rope_theta * self.alpha)
                ** (
                    torch.arange(0, self.spatial_dim, 2, device=device, dtype=dtype)
                    / self.spatial_dim
                )
            )
            h_angles = h_pos[:n_spatial].unsqueeze(-1) * spatial_freqs.unsqueeze(0)
            w_angles = w_pos[:n_spatial].unsqueeze(-1) * spatial_freqs.unsqueeze(0)

            spatial_cos_h[:n_spatial] = torch.cat(
                [h_angles.cos(), h_angles.cos()], dim=-1
            )
            spatial_sin_h[:n_spatial] = torch.cat(
                [h_angles.sin(), h_angles.sin()], dim=-1
            )
            spatial_cos_w[:n_spatial] = torch.cat(
                [w_angles.cos(), w_angles.cos()], dim=-1
            )
            spatial_sin_w[:n_spatial] = torch.cat(
                [w_angles.sin(), w_angles.sin()], dim=-1
            )

        # Passthrough dimensions: cos=1, sin=0
        pass_cos = torch.ones(
            seq_len, self.passthrough_dim, device=device, dtype=dtype
        )
        pass_sin = torch.zeros(
            seq_len, self.passthrough_dim, device=device, dtype=dtype
        )

        cos = torch.cat([seq_cos, spatial_cos_h, spatial_cos_w, pass_cos], dim=-1)
        sin = torch.cat([seq_sin, spatial_sin_h, spatial_sin_w, pass_sin], dim=-1)
        return cos.unsqueeze(0), sin.unsqueeze(0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class FeedForward(nn.Module):
    """Squared-ReLU gated feed-forward network.

    Gate and up projections are interleaved as [g0,u0,g1,u1,...] in the
    reference implementation. Here we use separate projections for clarity.
    Output = down_proj(relu(gate)^2 * up)
    """

    def __init__(self, config: FalconPerceptionConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_dim, config.ffn_hidden_dim, bias=False)
        self.up_proj = nn.Linear(config.hidden_dim, config.ffn_hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.ffn_hidden_dim, config.hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.relu(self.gate_proj(x))
        return self.down_proj(gate * gate * self.up_proj(x))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeat KV heads to match the number of query heads (for GQA).

    Args:
        x: (B, n_kv_heads, L, head_dim)
        n_rep: Number of times to repeat each KV head.

    Returns:
        (B, n_kv_heads * n_rep, L, head_dim)
    """
    if n_rep == 1:
        return x
    B, n_kv_heads, L, head_dim = x.shape
    x = x[:, :, None, :, :].expand(B, n_kv_heads, n_rep, L, head_dim)
    return x.reshape(B, n_kv_heads * n_rep, L, head_dim)


class HybridAttention(nn.Module):
    """Multi-head attention with hybrid masking and Grouped Query Attention.

    Image tokens attend bidirectionally to all tokens.
    Text/task tokens attend causally (only to previous tokens).
    Uses GQA: fewer KV heads than query heads (default 8 vs 16).
    Applies QK-norm (RMSNorm) to queries and keys before attention.
    """

    def __init__(self, config: FalconPerceptionConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_dim, q_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, config.hidden_dim, bias=False)

        # QK-norm: RMS normalization on Q and K for training stability
        self.q_norm = RMSNorm(self.head_dim, config.layer_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B, L, _ = x.shape
        q = self.q_proj(x).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # QK-norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Apply RoPE
        cos_q = cos[:, -L:, :].unsqueeze(1)
        sin_q = sin[:, -L:, :].unsqueeze(1)
        q = apply_rotary_embedding(q, cos_q, sin_q)
        # For GQA, RoPE is applied per KV head (same cos/sin, fewer heads)
        k = apply_rotary_embedding(k, cos_q, sin_q)

        # KV cache for autoregressive decoding
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_kv_cache = (k, v)

        # Expand KV heads to match query heads (GQA)
        k_expanded = repeat_kv(k, self.n_rep)
        v_expanded = repeat_kv(v, self.n_rep)

        # Scaled dot-product attention with mask
        scale = math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k_expanded.transpose(-2, -1)) / scale

        if attention_mask is not None:
            # attention_mask: (B, 1, L_q, L_kv) where False means masked
            attn_weights = attn_weights.masked_fill(~attention_mask, float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(
            q.dtype
        )
        attn_output = torch.matmul(attn_weights, v_expanded)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(attn_output), new_kv_cache


class TransformerBlock(nn.Module):
    def __init__(self, config: FalconPerceptionConfig):
        super().__init__()
        self.attention = HybridAttention(config)
        self.feed_forward = FeedForward(config)
        self.attention_norm = RMSNorm(config.hidden_dim, config.layer_norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_dim, config.layer_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        h = self.attention_norm(x)
        h, new_kv_cache = self.attention(h, cos, sin, attention_mask, kv_cache)
        x = x + h
        x = x + self.feed_forward(self.ffn_norm(x))
        return x, new_kv_cache


class BboxDecoder(nn.Module):
    """Two-layer MLP with squared-ReLU activation for bbox bin prediction.

    Matches the reference BboxDecoder: hidden_dim -> ffn -> relu^2 -> out_dim.
    Output is split in half for the two axes (x/y or w/h).
    """

    def __init__(self, config: FalconPerceptionConfig, out_bins: int):
        super().__init__()
        total_out = out_bins * 2  # Combined x+y or w+h
        ffn_dim = config.hidden_dim * 2
        self.fc1 = nn.Linear(config.hidden_dim, ffn_dim, bias=False)
        self.fc2 = nn.Linear(ffn_dim, total_out, bias=False)
        self.out_bins = out_bins

    def forward(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns logits for both axes. Input: (B, D), Output: (B, bins), (B, bins)."""
        h = F.relu(self.fc1(hidden_state))
        h = h * h  # squared ReLU
        logits = self.fc2(h)
        return logits[..., : self.out_bins], logits[..., self.out_bins :]


class CoordinateHead(BboxDecoder):
    """Predicts center (x, y) as two 1024-bin discrete classifications."""

    def __init__(self, config: FalconPerceptionConfig):
        super().__init__(config, config.coord_bins)


class SizeHead(BboxDecoder):
    """Predicts width/height as two 1024-bin discrete classifications (log-scale)."""

    def __init__(self, config: FalconPerceptionConfig):
        super().__init__(config, config.size_bins)


class AnyUpBlock(nn.Module):
    """Content-aware cross-attention upsampler block."""

    def __init__(self, in_dim: int, out_dim: int, num_heads: int = 4):
        super().__init__()
        self.q_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.k_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.v_proj = nn.Linear(in_dim, out_dim, bias=False)
        self.out_proj = nn.Linear(out_dim, out_dim, bias=False)
        self.norm = RMSNorm(out_dim)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.num_heads = num_heads
        self.head_dim = out_dim // num_heads

    def forward(
        self, low_res: torch.Tensor, high_res_query: torch.Tensor
    ) -> torch.Tensor:
        """Cross-attention upsampling.

        Args:
            low_res: (B, C_in, H, W) lower-resolution features
            high_res_query: (B, C_out, 2H, 2W) higher-resolution query features

        Returns:
            (B, C_out, 2H, 2W) upsampled features
        """
        B, C_in, H, W = low_res.shape
        _, C_out, H2, W2 = high_res_query.shape

        # Upsample low-res to match high-res spatial dims
        low_res_up = self.upsample(low_res)

        # Flatten spatial dims for attention
        kv = rearrange(low_res_up, "b c h w -> b (h w) c")
        q = rearrange(high_res_query, "b c h w -> b (h w) c")

        q = self.q_proj(q)
        k = self.k_proj(kv)
        v = self.v_proj(kv)

        # Multi-head attention
        q = rearrange(q, "b n (h d) -> b h n d", h=self.num_heads)
        k = rearrange(k, "b n (h d) -> b h n d", h=self.num_heads)
        v = rearrange(v, "b n (h d) -> b h n d", h=self.num_heads)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        out = rearrange(out, "b (h w) c -> b c h w", h=H2, w=W2)

        return self.norm(rearrange(out + high_res_query, "b c h w -> b h w c")).permute(
            0, 3, 1, 2
        )


class AnyUpUpsampler(nn.Module):
    """Multi-level AnyUp upsampler that produces high-resolution feature maps
    from intermediate transformer layer outputs for segmentation mask prediction."""

    def __init__(self, config: FalconPerceptionConfig):
        super().__init__()
        self.config = config
        hidden = config.anyup_hidden_dim
        # Project from transformer hidden dim to upsampler hidden dim
        self.input_proj = nn.Linear(config.hidden_dim, hidden)
        # Each level doubles spatial resolution
        self.levels = nn.ModuleList(
            [AnyUpBlock(hidden, hidden) for _ in range(config.anyup_levels)]
        )
        # Learnable query features for each upsampling level
        self.level_queries = nn.ParameterList(
            [nn.Parameter(torch.randn(1, hidden, 1, 1) * 0.02) for _ in range(config.anyup_levels)]
        )

    def forward(
        self,
        image_features: torch.Tensor,
        h_patches: int,
        w_patches: int,
    ) -> torch.Tensor:
        """Upsample image features from patch resolution to higher resolution.

        Args:
            image_features: (B, N_patches, D) transformer image token features
            h_patches: Number of patches in height
            w_patches: Number of patches in width

        Returns:
            (B, hidden_dim, H_up, W_up) upsampled features at 16x patch resolution
        """
        B = image_features.shape[0]
        x = self.input_proj(image_features)
        x = rearrange(x, "b (h w) c -> b c h w", h=h_patches, w=w_patches)

        for level, query_param in zip(self.levels, self.level_queries):
            _, _, H, W = x.shape
            query = query_param.expand(B, -1, H * 2, W * 2)
            x = level(x, query)

        return x


class SegmentationProjector(nn.Module):
    """Projects <seg> token hidden state for dot-product with upsampled features."""

    def __init__(self, config: FalconPerceptionConfig):
        super().__init__()
        self.proj = nn.Linear(config.hidden_dim, config.anyup_hidden_dim)

    def forward(self, seg_hidden: torch.Tensor) -> torch.Tensor:
        """Project segmentation token hidden state.

        Args:
            seg_hidden: (B, D) or (B, N, D)

        Returns:
            (B, anyup_hidden_dim) or (B, N, anyup_hidden_dim)
        """
        return self.proj(seg_hidden)


class FalconPerceptionModel(nn.Module):
    """Falcon Perception: unified dense Transformer for open-vocabulary
    object detection and instance segmentation.

    600M parameters, early-fusion architecture processing image patches
    and text tokens in a shared parameter space from layer 1.
    """

    def __init__(self, config: FalconPerceptionConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_dim)

        # Patch embedding for image tokens
        self.patch_embed = nn.Conv2d(
            3,
            config.hidden_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=True,
        )

        # Positional encoding
        self.rope = GoldenGateRoPE(config)

        # Transformer layers
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_dim, config.layer_norm_eps)

        # Specialized heads
        self.coord_head = CoordinateHead(config)
        self.size_head = SizeHead(config)
        self.seg_projector = SegmentationProjector(config)

        # AnyUp upsampler for segmentation
        self.anyup = AnyUpUpsampler(config)

        # LM head for text token prediction (presence/absence, eoq, eos)
        self.lm_head = nn.Linear(config.hidden_dim, config.vocab_size, bias=False)

        # Fourier feature projection for re-injection of coord/size predictions
        self.coord_fourier_proj = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False
        )
        self.size_fourier_proj = nn.Linear(
            config.hidden_dim, config.hidden_dim, bias=False
        )

    def embed_image(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.

        Args:
            pixel_values: (B, 3, H, W) normalized image tensor

        Returns:
            (B, N_patches, D) patch embeddings
        """
        patches = self.patch_embed(pixel_values)  # (B, D, H/P, W/P)
        B, D, H, W = patches.shape
        return rearrange(patches, "b d h w -> b (h w) d"), H, W

    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Embed text/special token IDs.

        Args:
            token_ids: (B, L) integer token IDs

        Returns:
            (B, L, D) token embeddings
        """
        return self.token_embedding(token_ids)

    def build_hybrid_attention_mask(
        self,
        batch_size: int,
        seq_len: int,
        num_image_tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build hybrid attention mask.

        Image tokens (positions 0..num_image_tokens-1) attend bidirectionally
        to all tokens. Text/task tokens attend causally.

        Returns:
            (B, 1, seq_len, seq_len) boolean mask (True = attend, False = masked).
        """
        # Start with causal mask
        mask = torch.tril(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool)
        )

        # Image tokens can attend to all positions (bidirectional)
        mask[:num_image_tokens, :] = True

        # All tokens can attend to image tokens
        mask[:, :num_image_tokens] = True

        return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)

    def forward_transformer(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run hidden states through all transformer layers.

        Returns:
            (hidden_states, new_kv_caches)
        """
        new_kv_caches = []
        for i, layer in enumerate(self.layers):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            hidden_states, new_kv = layer(
                hidden_states, cos, sin, attention_mask, kv_cache
            )
            new_kv_caches.append(new_kv)
        hidden_states = self.norm(hidden_states)
        return hidden_states, new_kv_caches

    def predict_next_special_token(
        self, hidden_state: torch.Tensor
    ) -> torch.Tensor:
        """Predict next token logits from the last hidden state.

        Args:
            hidden_state: (B, D) last position hidden state

        Returns:
            (B, vocab_size) logits
        """
        return self.lm_head(hidden_state)

    def predict_coord(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict coordinate bins from <coord> token hidden state.

        Returns:
            (x_logits, y_logits) each (B, coord_bins)
        """
        return self.coord_head(hidden_state)

    def predict_size(
        self, hidden_state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predict size bins from <size> token hidden state.

        Returns:
            (w_logits, h_logits) each (B, size_bins)
        """
        return self.size_head(hidden_state)

    def get_seg_projection(self, hidden_state: torch.Tensor) -> torch.Tensor:
        """Project <seg> token hidden state for mask computation.

        Args:
            hidden_state: (B, D)

        Returns:
            (B, anyup_hidden_dim)
        """
        return self.seg_projector(hidden_state)

    def compute_mask(
        self,
        seg_projection: torch.Tensor,
        upsampled_features: torch.Tensor,
    ) -> torch.Tensor:
        """Compute binary mask from seg projection and upsampled image features.

        Args:
            seg_projection: (B, anyup_hidden_dim) from <seg> token
            upsampled_features: (B, anyup_hidden_dim, H, W) from AnyUp

        Returns:
            (B, H, W) mask logits (before sigmoid)
        """
        # Dot product between seg projection and spatial features
        seg_proj = seg_projection.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
        return (upsampled_features * seg_proj).sum(dim=1)  # (B, H, W)
