from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from inference_models.errors import ModelInputError

DEFAULT_MAX_TEXT_LINE_WIDTH = 3200


def preprocess_text_lines(
    images: List[np.ndarray],
    target_height: int,
    min_width: int,
    max_width: int = DEFAULT_MAX_TEXT_LINE_WIDTH,
) -> np.ndarray:
    max_wh_ratio = min_width / float(target_height)
    for image in images:
        height, width = image.shape[:2]
        if height <= 0 or width <= 0:
            raise ModelInputError(
                message="Cannot run PP-OCRv6 recognition on an empty image.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        max_wh_ratio = max(max_wh_ratio, width / float(height))
    target_width = int(np.ceil(target_height * max_wh_ratio))
    target_width = min(max(target_width, min_width), max_width)
    return np.stack(
        [
            resize_and_pad_text_line(
                image=image,
                target_height=target_height,
                target_width=target_width,
            )
            for image in images
        ],
        axis=0,
    )


def resize_and_pad_text_line(
    image: np.ndarray, target_height: int, target_width: int
) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= 0 or width <= 0:
        raise ModelInputError(
            message="Cannot run PP-OCRv6 recognition on an empty image.",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    resized_width = int(np.ceil(target_height * width / float(height)))
    resized_width = max(1, min(resized_width, target_width))
    resized = cv2.resize(image, (resized_width, target_height))
    normalized = (resized.astype("float32") / 255.0 - 0.5) / 0.5
    normalized = np.transpose(normalized, (2, 0, 1))
    padded = np.zeros((image.shape[2], target_height, target_width), dtype="float32")
    padded[:, :, :resized_width] = normalized
    return padded


def preprocess_text_lines_torch(
    images: List[torch.Tensor],
    target_height: int,
    min_width: int,
    max_width: int = DEFAULT_MAX_TEXT_LINE_WIDTH,
) -> torch.Tensor:
    """Device-native counterpart of ``preprocess_text_lines``.

    ``images`` are ``CHW`` float tensors in ``[0, 255]`` on the model device.
    Returns the normalized, right-zero-padded ``NCHW`` batch on the same device,
    with the batch width derived from the widest text-line aspect ratio (as in
    the numpy path), without a numpy/cv2 round-trip.
    """
    max_wh_ratio = min_width / float(target_height)
    for image in images:
        _, height, width = image.shape
        if height <= 0 or width <= 0:
            raise ModelInputError(
                message="Cannot run PP-OCRv6 recognition on an empty image.",
                help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
            )
        max_wh_ratio = max(max_wh_ratio, width / float(height))
    target_width = int(np.ceil(target_height * max_wh_ratio))
    target_width = min(max(target_width, min_width), max_width)
    processed = []
    for image in images:
        channels, height, width = image.shape
        resized_width = int(np.ceil(target_height * width / float(height)))
        resized_width = max(1, min(resized_width, target_width))
        resized = F.interpolate(
            image.unsqueeze(0),
            size=(target_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )[0]
        normalized = (resized / 255.0 - 0.5) / 0.5
        padded = torch.zeros(
            (channels, target_height, target_width),
            dtype=normalized.dtype,
            device=normalized.device,
        )
        padded[:, :, :resized_width] = normalized
        processed.append(padded)
    return torch.stack(processed, dim=0)


def ctc_decode(
    predictions: np.ndarray, characters: List[str]
) -> List[Tuple[str, float]]:
    predictions_idx = predictions.argmax(axis=2)
    predictions_prob = predictions.max(axis=2)
    return ctc_decode_indices(
        indices=predictions_idx,
        probs=predictions_prob,
        characters=characters,
    )


def ctc_decode_indices(
    indices: np.ndarray, probs: np.ndarray, characters: List[str]
) -> List[Tuple[str, float]]:
    blank_idx = 0
    character_by_idx = [""] + characters + [" "]
    decoded = []
    for batch_idx, sequence in enumerate(indices):
        tokens = []
        token_scores = []
        previous_idx = None
        for step_idx, token_idx in enumerate(sequence):
            token_idx = int(token_idx)
            if token_idx == blank_idx or token_idx == previous_idx:
                previous_idx = token_idx
                continue
            if token_idx < len(character_by_idx):
                tokens.append(character_by_idx[token_idx])
                token_scores.append(float(probs[batch_idx][step_idx]))
            previous_idx = token_idx
        score = float(np.mean(token_scores)) if token_scores else 0.0
        decoded.append(("".join(tokens), score))
    return decoded


def load_inference_config(config_path: str) -> Tuple[Tuple[int, int, int], List[str]]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    image_shape = _extract_image_shape(config)
    characters = _extract_characters(config)

    if image_shape is None:
        raise ModelInputError(
            message=f"Could not find image_shape in PP-OCRv6 config: {config_path}",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    if not characters:
        raise ModelInputError(
            message=f"Could not find character_dict in PP-OCRv6 config: {config_path}",
            help_url="https://inference-models.roboflow.com/errors/input-validation/#modelinputerror",
        )
    return image_shape, characters


def _extract_image_shape(config: Any) -> Optional[Tuple[int, int, int]]:
    if not isinstance(config, dict):
        return None
    transform_ops = (config.get("PreProcess") or {}).get("transform_ops") or []
    for op in transform_ops:
        if not isinstance(op, dict) or "RecResizeImg" not in op:
            continue
        image_shape = (op["RecResizeImg"] or {}).get("image_shape")
        if image_shape and len(image_shape) == 3:
            return tuple(int(value) for value in image_shape)
    return None


def _extract_characters(config: Any) -> List[str]:
    if not isinstance(config, dict):
        return []
    character_dict = (config.get("PostProcess") or {}).get("character_dict") or []
    return [str(character) for character in character_dict]
