# Copyright 2026 Technology Innovation Institute (TII), Abu Dhabi.
# Licensed under the Apache License, Version 2.0.
# Adapted from https://github.com/tiiuae/Falcon-Perception for integration
# with the inference-models package.

from dataclasses import dataclass


@dataclass(frozen=True)
class FalconPerceptionConfig:
    """Configuration for the Falcon Perception 600M model.

    Default values match the reference tiiuae/Falcon-Perception checkpoint:
    28 layers, 1024 hidden dim, 16 query heads / 8 KV heads (GQA),
    head_dim=128, vocab_size=65536, squared-ReLU gated FFN.
    """

    hidden_dim: int = 1024
    num_heads: int = 16
    num_kv_heads: int = 8  # Grouped Query Attention: fewer KV heads than Q heads
    head_dim: int = 128  # Explicit head dim (not hidden_dim // num_heads)
    num_layers: int = 28
    ffn_hidden_dim: int = 3072
    vocab_size: int = 65536
    patch_size: int = 16
    max_image_size: int = 1024
    max_seq_len: int = 8192
    coord_bins: int = 1024  # Per axis (total coord_out_dim = 2048, split x/y)
    size_bins: int = 1024  # Per axis (total size_out_dim = 2048, split w/h)
    seg_dim: int = 256  # segm_out_dim for mask projection
    anyup_levels: int = 4
    anyup_hidden_dim: int = 256
    dropout: float = 0.0
    layer_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    gg_rope_alpha: float = 1.618033988749895  # Golden ratio for GGRoPE
    log2_size_range: float = 10.0  # log2(max_size/min_size) for size decoding
    mask_threshold: float = 0.5
    max_instances_per_query: int = 256
    max_generation_tokens: int = 2048

    # Image normalization (Falcon Perception uses 0.5/0.5 not ImageNet)
    image_mean: tuple = (0.5, 0.5, 0.5)
    image_std: tuple = (0.5, 0.5, 0.5)

    # Special token IDs (set during tokenizer initialization)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    eoq_token_id: int = 3
    present_token_id: int = 4
    absent_token_id: int = 5
    coord_token_id: int = 6
    size_token_id: int = 7
    seg_token_id: int = 8
    image_token_id: int = 9


DEFAULT_CONFIG = FalconPerceptionConfig()
