# Copyright 2026 Technology Innovation Institute (TII), Abu Dhabi.
# Licensed under the Apache License, Version 2.0.
# Adapted from https://github.com/tiiuae/Falcon-Perception for integration
# with the inference-models package.
#
# Image preprocessing and tokenization for Falcon Perception.

import json
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import torch
from tokenizers import Tokenizer

from inference_models.models.falcon_perception.config import FalconPerceptionConfig

# Falcon Perception uses 0.5/0.5 normalization (not ImageNet stats)
DEFAULT_IMAGE_MEAN = [0.5, 0.5, 0.5]
DEFAULT_IMAGE_STD = [0.5, 0.5, 0.5]


@dataclass
class ImageMetadata:
    """Metadata about a preprocessed image for coordinate rescaling."""

    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    h_patches: int
    w_patches: int
    pad_h: int
    pad_w: int


def load_tokenizer(
    tokenizer_path: str, config: FalconPerceptionConfig
) -> Tokenizer:
    """Load a BPE tokenizer from a tokenizer.json file.

    Args:
        tokenizer_path: Path to tokenizer.json file.
        config: Model config (for special token IDs).

    Returns:
        Initialized Tokenizer instance.
    """
    tokenizer = Tokenizer.from_file(tokenizer_path)
    return tokenizer


def get_special_token_ids(
    tokenizer_path: str,
) -> dict:
    """Extract special token IDs from the tokenizer config.

    Reads the tokenizer.json to find IDs for special tokens like
    <present>, <absent>, <coord>, <size>, <seg>, <eoq>, <eos>.

    Args:
        tokenizer_path: Path to tokenizer.json file.

    Returns:
        Dict mapping token names to their IDs.
    """
    with open(tokenizer_path, "r") as f:
        tokenizer_config = json.load(f)

    # Build reverse lookup from token string to ID
    added_tokens = tokenizer_config.get("added_tokens", [])
    token_to_id = {}
    for token_entry in added_tokens:
        token_to_id[token_entry["content"]] = token_entry["id"]

    special_names = {
        "pad_token_id": "<pad>",
        "bos_token_id": "<bos>",
        "eos_token_id": "<eos>",
        "eoq_token_id": "<eoq>",
        "present_token_id": "<present>",
        "absent_token_id": "<absent>",
        "coord_token_id": "<coord>",
        "size_token_id": "<size>",
        "seg_token_id": "<seg>",
        "image_token_id": "<image>",
    }

    result = {}
    for field_name, token_str in special_names.items():
        if token_str in token_to_id:
            result[field_name] = token_to_id[token_str]
    return result


def resize_image_preserve_aspect(
    image: np.ndarray, max_size: int
) -> Tuple[np.ndarray, int, int]:
    """Resize image preserving aspect ratio to fit within max_size x max_size.

    Args:
        image: (H, W, 3) uint8 RGB image.
        max_size: Maximum dimension.

    Returns:
        (resized_image, new_h, new_w)
    """
    h, w = image.shape[:2]
    if max(h, w) <= max_size:
        return image, h, w

    scale = max_size / max(h, w)
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Use PIL-style bilinear resize via numpy/torch
    image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    resized = torch.nn.functional.interpolate(
        image_tensor,
        size=(new_h, new_w),
        mode="bilinear",
        align_corners=False,
    )
    return (
        resized.squeeze(0).permute(1, 2, 0).to(torch.uint8).numpy(),
        new_h,
        new_w,
    )


def pad_to_patch_multiple(
    image: np.ndarray, patch_size: int
) -> Tuple[np.ndarray, int, int]:
    """Pad image so dimensions are multiples of patch_size.

    Args:
        image: (H, W, 3) uint8 image.
        patch_size: Patch dimension.

    Returns:
        (padded_image, pad_h, pad_w)
    """
    h, w = image.shape[:2]
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        image = np.pad(
            image, ((0, pad_h), (0, pad_w), (0, 0)), mode="constant", constant_values=0
        )
    return image, pad_h, pad_w


def normalize_image(
    image: np.ndarray,
    mean: tuple = None,
    std: tuple = None,
) -> torch.Tensor:
    """Normalize image and convert to tensor.

    Args:
        image: (H, W, 3) uint8 RGB image.
        mean: Per-channel mean (default: 0.5, 0.5, 0.5).
        std: Per-channel std (default: 0.5, 0.5, 0.5).

    Returns:
        (3, H, W) float32 tensor, normalized.
    """
    if mean is None:
        mean = DEFAULT_IMAGE_MEAN
    if std is None:
        std = DEFAULT_IMAGE_STD
    tensor = torch.from_numpy(image).float() / 255.0
    tensor = tensor.permute(2, 0, 1)  # (3, H, W)
    mean_t = torch.tensor(list(mean), dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(list(std), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def preprocess_image(
    image: np.ndarray,
    config: FalconPerceptionConfig,
) -> Tuple[torch.Tensor, ImageMetadata]:
    """Full image preprocessing pipeline.

    1. Resize preserving aspect ratio to max_image_size
    2. Pad to patch_size multiple
    3. Normalize to ImageNet stats

    Args:
        image: (H, W, 3) uint8 RGB image.
        config: Model configuration.

    Returns:
        (pixel_values, metadata) where pixel_values is (3, H_padded, W_padded).
    """
    original_h, original_w = image.shape[:2]
    resized, new_h, new_w = resize_image_preserve_aspect(image, config.max_image_size)
    padded, pad_h, pad_w = pad_to_patch_multiple(resized, config.patch_size)
    pixel_values = normalize_image(padded, mean=config.image_mean, std=config.image_std)

    h_patches = (new_h + pad_h) // config.patch_size
    w_patches = (new_w + pad_w) // config.patch_size

    metadata = ImageMetadata(
        original_height=original_h,
        original_width=original_w,
        resized_height=new_h,
        resized_width=new_w,
        h_patches=h_patches,
        w_patches=w_patches,
        pad_h=pad_h,
        pad_w=pad_w,
    )
    return pixel_values, metadata


def tokenize_prompts(
    prompts: List[str],
    tokenizer: Tokenizer,
    config: FalconPerceptionConfig,
) -> torch.Tensor:
    """Tokenize text prompts into token IDs.

    Each prompt is tokenized and separated by <eoq> tokens.
    The full sequence is: <bos> prompt1_tokens <eoq> prompt2_tokens <eoq> ... <eos>
    (But <eos> is generated by the model, not included in input.)

    Args:
        prompts: List of text prompt strings.
        tokenizer: BPE tokenizer.
        config: Model configuration.

    Returns:
        (L,) tensor of token IDs.
    """
    all_ids = [config.bos_token_id]
    for i, prompt in enumerate(prompts):
        encoding = tokenizer.encode(prompt)
        all_ids.extend(encoding.ids)
        all_ids.append(config.eoq_token_id)
    return torch.tensor(all_ids, dtype=torch.long)
