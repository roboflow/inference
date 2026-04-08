# Copyright 2026 Technology Innovation Institute (TII), Abu Dhabi.
# Licensed under the Apache License, Version 2.0.
# Adapted from https://github.com/tiiuae/Falcon-Perception for integration
# with the inference-models package.
#
# Batch inference engine for Falcon Perception.
# Left-pads sequences, runs single prefill pass, then token-by-token
# autoregressive decoding with dense KV cache.

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

from inference_models.models.falcon_perception.config import FalconPerceptionConfig
from inference_models.models.falcon_perception.model import (
    FalconPerceptionModel,
    build_fourier_features,
)
from inference_models.models.falcon_perception.preprocessing import ImageMetadata


@dataclass
class InstancePrediction:
    """A single detected instance from the model."""

    center_x_bin: int
    center_y_bin: int
    width_bin: int
    height_bin: int
    coord_x_confidence: float
    coord_y_confidence: float
    size_w_confidence: float
    size_h_confidence: float
    seg_hidden: Optional[torch.Tensor] = None  # (D,) hidden state for mask


@dataclass
class QueryResult:
    """Result for a single text prompt/query."""

    prompt: str
    present: bool
    presence_confidence: float
    instances: List[InstancePrediction] = field(default_factory=list)


@dataclass
class BatchInferenceResult:
    """Result for a single image in the batch."""

    query_results: List[QueryResult]
    image_features: Optional[torch.Tensor] = None  # For mask computation
    h_patches: int = 0
    w_patches: int = 0


def decode_coord_bins(
    x_bin: int, y_bin: int, image_width: int, image_height: int, num_bins: int = 1024
) -> Tuple[float, float]:
    """Convert coordinate bin indices to pixel coordinates.

    Args:
        x_bin: X center bin index (0 to num_bins-1).
        y_bin: Y center bin index (0 to num_bins-1).
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        num_bins: Number of discretization bins.

    Returns:
        (center_x, center_y) in pixel coordinates.
    """
    center_x = x_bin / (num_bins - 1) * image_width
    center_y = y_bin / (num_bins - 1) * image_height
    return center_x, center_y


def decode_size_bins(
    w_bin: int, h_bin: int, image_width: int, image_height: int,
    num_bins: int = 1024, log2_range: float = 10.0,
) -> Tuple[float, float]:
    """Convert size bin indices to pixel dimensions using log-scale decoding.

    The log-scale mapping: size = 2^(bin_value / (num_bins-1) * log2_range)
    This is then scaled relative to image dimensions.

    Args:
        w_bin: Width bin index (0 to num_bins-1).
        h_bin: Height bin index (0 to num_bins-1).
        image_width: Original image width in pixels.
        image_height: Original image height in pixels.
        num_bins: Number of discretization bins.
        log2_range: log2 range for the size mapping.

    Returns:
        (width, height) in pixel dimensions.
    """
    # Normalized size in [0, 1] mapped through log scale
    w_norm = 2.0 ** (w_bin / (num_bins - 1) * log2_range) / (2.0**log2_range)
    h_norm = 2.0 ** (h_bin / (num_bins - 1) * log2_range) / (2.0**log2_range)
    return w_norm * image_width, h_norm * image_height


def boxes_from_instances(
    instances: List[InstancePrediction],
    image_width: int,
    image_height: int,
    config: FalconPerceptionConfig,
) -> List[Tuple[float, float, float, float]]:
    """Convert instance predictions to (x1, y1, x2, y2) bounding boxes.

    Returns list of bounding boxes in pixel coordinates, clipped to image bounds.
    """
    boxes = []
    for inst in instances:
        cx, cy = decode_coord_bins(
            inst.center_x_bin,
            inst.center_y_bin,
            image_width,
            image_height,
            config.coord_bins,
        )
        w, h = decode_size_bins(
            inst.width_bin,
            inst.height_bin,
            image_width,
            image_height,
            config.size_bins,
            config.log2_size_range,
        )
        x1 = max(0.0, cx - w / 2)
        y1 = max(0.0, cy - h / 2)
        x2 = min(float(image_width), cx + w / 2)
        y2 = min(float(image_height), cy + h / 2)
        boxes.append((x1, y1, x2, y2))
    return boxes


def instance_confidence(inst: InstancePrediction) -> float:
    """Compute per-instance confidence from coordinate and size head logit maxes.

    Geometric mean of the four head confidences (coord_x, coord_y, size_w, size_h).
    """
    return (
        inst.coord_x_confidence
        * inst.coord_y_confidence
        * inst.size_w_confidence
        * inst.size_h_confidence
    ) ** 0.25


class BatchEngine:
    """Batch inference engine for Falcon Perception.

    Implements the simplest inference path:
    1. Left-pad all sequences to the same length
    2. Single prefill pass for image + text tokens
    3. Token-by-token autoregressive decode with dense KV cache
    4. Generate until all sequences emit <eos>
    """

    def __init__(
        self,
        model: FalconPerceptionModel,
        config: FalconPerceptionConfig,
        device: torch.device,
    ):
        self.model = model
        self.config = config
        self.device = device

    @torch.inference_mode()
    def run(
        self,
        pixel_values: torch.Tensor,
        text_token_ids: torch.Tensor,
        image_metadata: ImageMetadata,
        prompts: List[str],
        task: str = "segmentation",
    ) -> BatchInferenceResult:
        """Run batch inference for a single image with multiple prompts.

        Args:
            pixel_values: (3, H, W) preprocessed image tensor
            text_token_ids: (L_text,) token IDs for all prompts
            image_metadata: Preprocessing metadata
            prompts: Original prompt strings
            task: "detection" or "segmentation"

        Returns:
            BatchInferenceResult with per-query results.
        """
        config = self.config

        # 1. Embed image patches
        pixel_values = pixel_values.unsqueeze(0).to(self.device)  # (1, 3, H, W)
        image_embeds, h_patches, w_patches = self.model.embed_image(pixel_values)
        num_image_tokens = image_embeds.shape[1]

        # 2. Embed text tokens
        text_token_ids = text_token_ids.unsqueeze(0).to(self.device)  # (1, L_text)
        text_embeds = self.model.embed_tokens(text_token_ids)

        # 3. Concatenate: [image_tokens, text_tokens]
        input_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        seq_len = input_embeds.shape[1]

        # 4. Build RoPE and attention mask for prefill
        cos, sin = self.model.rope(
            seq_len=seq_len,
            image_h_patches=h_patches,
            image_w_patches=w_patches,
            num_image_tokens=num_image_tokens,
            device=self.device,
            dtype=input_embeds.dtype,
        )
        attention_mask = self.model.build_hybrid_attention_mask(
            batch_size=1,
            seq_len=seq_len,
            num_image_tokens=num_image_tokens,
            device=self.device,
            dtype=input_embeds.dtype,
        )

        # 5. Prefill: run through all transformer layers
        hidden_states, kv_caches = self.model.forward_transformer(
            input_embeds, cos, sin, attention_mask
        )

        # Save image features for segmentation
        image_features = hidden_states[:, :num_image_tokens, :]

        # 6. Autoregressive decode
        query_results = self._decode_loop(
            hidden_states=hidden_states,
            kv_caches=kv_caches,
            seq_len=seq_len,
            num_image_tokens=num_image_tokens,
            h_patches=h_patches,
            w_patches=w_patches,
            prompts=prompts,
            task=task,
        )

        return BatchInferenceResult(
            query_results=query_results,
            image_features=image_features if task == "segmentation" else None,
            h_patches=h_patches,
            w_patches=w_patches,
        )

    def _get_kv_len(
        self, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]]
    ) -> int:
        """Get the current KV cache sequence length."""
        return kv_caches[0][0].shape[2]

    def _step(
        self,
        embed: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        num_image_tokens: int,
        h_patches: int,
        w_patches: int,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """Run a single decode step: forward one token through the transformer.

        Builds the correct RoPE and attention mask based on current KV cache length.

        Args:
            embed: (1, 1, D) token embedding for the new token.
            kv_caches: Current KV caches from previous steps.
            num_image_tokens: Number of image tokens (for RoPE).
            h_patches, w_patches: Image patch grid dimensions (for RoPE).

        Returns:
            (last_hidden, updated_kv_caches)
        """
        kv_len = self._get_kv_len(kv_caches)
        new_total_len = kv_len + 1

        # Attention mask: new token can attend to all previous tokens + itself
        attn_mask = torch.ones(
            1, 1, 1, new_total_len, device=self.device, dtype=torch.bool
        )

        # RoPE: compute for the full sequence length so the position encoding is correct
        cos, sin = self.model.rope(
            seq_len=new_total_len,
            image_h_patches=h_patches,
            image_w_patches=w_patches,
            num_image_tokens=num_image_tokens,
            device=self.device,
            dtype=embed.dtype,
        )

        return self.model.forward_transformer(
            embed, cos, sin, attn_mask, kv_caches
        )

    def _decode_loop(
        self,
        hidden_states: torch.Tensor,
        kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
        seq_len: int,
        num_image_tokens: int,
        h_patches: int,
        w_patches: int,
        prompts: List[str],
        task: str,
    ) -> List[QueryResult]:
        """Autoregressive decoding loop.

        Generates tokens one at a time, interpreting special tokens
        to build structured outputs (coord, size, seg for each instance).
        """
        config = self.config

        # Get last hidden state for initial prediction
        last_hidden = hidden_states[:, -1:, :]  # (1, 1, D)

        query_results: List[QueryResult] = []
        current_prompt_idx = 0
        current_instances: List[InstancePrediction] = []
        steps = 0

        # State machine for decoding
        state = "expect_presence"
        current_instance: Optional[InstancePrediction] = None
        presence_conf = 0.0

        while steps < config.max_generation_tokens:
            steps += 1

            if state == "expect_presence":
                # Predict presence/absence token
                logits = self.model.predict_next_special_token(
                    last_hidden[:, -1, :]
                )
                presence_logits = logits[
                    :, [config.present_token_id, config.absent_token_id]
                ]
                presence_probs = F.softmax(presence_logits, dim=-1)
                is_present = presence_probs[0, 0] > presence_probs[0, 1]
                presence_conf = presence_probs[0, 0].item()

                if is_present:
                    next_token_id = config.present_token_id
                    state = "expect_coord_or_eoq"
                else:
                    next_token_id = config.absent_token_id
                    query_results.append(
                        QueryResult(
                            prompt=prompts[current_prompt_idx]
                            if current_prompt_idx < len(prompts)
                            else "",
                            present=False,
                            presence_confidence=1.0 - presence_conf,
                            instances=[],
                        )
                    )
                    current_prompt_idx += 1
                    if current_prompt_idx >= len(prompts):
                        break
                    state = "expect_presence"

                next_embed = self.model.embed_tokens(
                    torch.tensor([[next_token_id]], device=self.device)
                )
                last_hidden, kv_caches = self._step(
                    next_embed, kv_caches, num_image_tokens, h_patches, w_patches
                )

            elif state == "expect_coord_or_eoq":
                logits = self.model.predict_next_special_token(
                    last_hidden[:, -1, :]
                )
                coord_eoq_logits = logits[
                    :, [config.coord_token_id, config.eoq_token_id, config.eos_token_id]
                ]
                coord_eoq_probs = F.softmax(coord_eoq_logits, dim=-1)
                best_idx = coord_eoq_probs.argmax(dim=-1).item()

                if best_idx == 1:  # <eoq>
                    query_results.append(
                        QueryResult(
                            prompt=prompts[current_prompt_idx]
                            if current_prompt_idx < len(prompts)
                            else "",
                            present=True,
                            presence_confidence=presence_conf,
                            instances=list(current_instances),
                        )
                    )
                    current_instances = []
                    current_prompt_idx += 1
                    if current_prompt_idx >= len(prompts):
                        break
                    next_token_id = config.eoq_token_id
                    state = "expect_presence"

                    next_embed = self.model.embed_tokens(
                        torch.tensor([[next_token_id]], device=self.device)
                    )
                    last_hidden, kv_caches = self._step(
                        next_embed, kv_caches, num_image_tokens, h_patches, w_patches
                    )

                elif best_idx == 2:  # <eos>
                    if current_instances:
                        query_results.append(
                            QueryResult(
                                prompt=prompts[current_prompt_idx]
                                if current_prompt_idx < len(prompts)
                                else "",
                                present=True,
                                presence_confidence=presence_conf,
                                instances=list(current_instances),
                            )
                        )
                    for i in range(current_prompt_idx + 1, len(prompts)):
                        query_results.append(
                            QueryResult(
                                prompt=prompts[i],
                                present=False,
                                presence_confidence=0.0,
                                instances=[],
                            )
                        )
                    break

                else:  # <coord>
                    next_embed = self.model.embed_tokens(
                        torch.tensor([[config.coord_token_id]], device=self.device)
                    )
                    last_hidden, kv_caches = self._step(
                        next_embed, kv_caches, num_image_tokens, h_patches, w_patches
                    )
                    state = "decode_coord"

            elif state == "decode_coord":
                x_logits, y_logits = self.model.predict_coord(
                    last_hidden[:, -1, :]
                )
                x_bin = x_logits.argmax(dim=-1).item()
                y_bin = y_logits.argmax(dim=-1).item()
                x_conf = F.softmax(x_logits, dim=-1).max(dim=-1).values.item()
                y_conf = F.softmax(y_logits, dim=-1).max(dim=-1).values.item()

                current_instance = InstancePrediction(
                    center_x_bin=x_bin,
                    center_y_bin=y_bin,
                    width_bin=0,
                    height_bin=0,
                    coord_x_confidence=x_conf,
                    coord_y_confidence=y_conf,
                    size_w_confidence=0.0,
                    size_h_confidence=0.0,
                )

                # Re-inject coord as Fourier features with <size> token
                x_norm = torch.tensor(
                    [[x_bin / max(config.coord_bins - 1, 1)]],
                    device=self.device,
                    dtype=last_hidden.dtype,
                )
                y_norm = torch.tensor(
                    [[y_bin / max(config.coord_bins - 1, 1)]],
                    device=self.device,
                    dtype=last_hidden.dtype,
                )
                coord_values = torch.cat([x_norm, y_norm], dim=-1)
                fourier_feats = build_fourier_features(
                    coord_values.unsqueeze(0), config.hidden_dim // 2
                )
                fourier_feats = fourier_feats.reshape(1, -1)
                coord_inject = self.model.coord_fourier_proj(
                    fourier_feats.unsqueeze(1)
                )

                size_embed = self.model.embed_tokens(
                    torch.tensor([[config.size_token_id]], device=self.device)
                )
                combined = size_embed + coord_inject
                last_hidden, kv_caches = self._step(
                    combined, kv_caches, num_image_tokens, h_patches, w_patches
                )
                steps += 1
                state = "decode_size"

            elif state == "decode_size":
                w_logits, h_logits = self.model.predict_size(
                    last_hidden[:, -1, :]
                )
                w_bin = w_logits.argmax(dim=-1).item()
                h_bin = h_logits.argmax(dim=-1).item()
                w_conf = F.softmax(w_logits, dim=-1).max(dim=-1).values.item()
                h_conf = F.softmax(h_logits, dim=-1).max(dim=-1).values.item()

                current_instance.width_bin = w_bin
                current_instance.height_bin = h_bin
                current_instance.size_w_confidence = w_conf
                current_instance.size_h_confidence = h_conf

                if task == "segmentation":
                    # Re-inject size as Fourier features with <seg> token
                    w_norm = torch.tensor(
                        [[w_bin / max(config.size_bins - 1, 1)]],
                        device=self.device,
                        dtype=last_hidden.dtype,
                    )
                    h_norm = torch.tensor(
                        [[h_bin / max(config.size_bins - 1, 1)]],
                        device=self.device,
                        dtype=last_hidden.dtype,
                    )
                    size_values = torch.cat([w_norm, h_norm], dim=-1)
                    fourier_feats = build_fourier_features(
                        size_values.unsqueeze(0), config.hidden_dim // 2
                    )
                    fourier_feats = fourier_feats.reshape(1, -1)
                    size_inject = self.model.size_fourier_proj(
                        fourier_feats.unsqueeze(1)
                    )

                    seg_embed = self.model.embed_tokens(
                        torch.tensor([[config.seg_token_id]], device=self.device)
                    )
                    combined = seg_embed + size_inject
                    last_hidden, kv_caches = self._step(
                        combined, kv_caches, num_image_tokens, h_patches, w_patches
                    )
                    steps += 1

                    current_instance.seg_hidden = last_hidden[:, -1, :].detach()

                current_instances.append(current_instance)
                current_instance = None
                state = "expect_coord_or_eoq"

        # Handle case where generation ended without emitting all queries
        while len(query_results) < len(prompts):
            idx = len(query_results)
            query_results.append(
                QueryResult(
                    prompt=prompts[idx] if idx < len(prompts) else "",
                    present=False,
                    presence_confidence=0.0,
                    instances=[],
                )
            )

        return query_results
