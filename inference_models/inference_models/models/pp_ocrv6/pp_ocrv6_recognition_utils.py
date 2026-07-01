import ast
from typing import List, Tuple

import cv2
import numpy as np

from inference_models.errors import ModelInputError


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
    padded = np.zeros((target_height, target_width, image.shape[2]), dtype=np.uint8)
    padded[:, :resized_width, :] = resized
    normalized = padded.astype("float32") / 255.0
    normalized = (normalized - 0.5) / 0.5
    return np.transpose(normalized, (2, 0, 1))


def ctc_decode(
    predictions: np.ndarray, characters: List[str]
) -> List[Tuple[str, float]]:
    blank_idx = 0
    character_by_idx = [""] + characters + [" "]
    predictions_idx = predictions.argmax(axis=2)
    predictions_prob = predictions.max(axis=2)
    decoded = []
    for batch_idx, sequence in enumerate(predictions_idx):
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
                token_scores.append(float(predictions_prob[batch_idx][step_idx]))
            previous_idx = token_idx
        score = float(np.mean(token_scores)) if token_scores else 0.0
        decoded.append(("".join(tokens), score))
    return decoded


def load_inference_config(config_path: str) -> Tuple[Tuple[int, int, int], List[str]]:
    image_shape = None
    characters = []
    with open(config_path, "r", encoding="utf-8") as config_file:
        lines = config_file.readlines()

    for idx, line in enumerate(lines):
        if line.strip() == "image_shape:":
            image_shape = (
                int(lines[idx + 1].strip().lstrip("-").strip()),
                int(lines[idx + 2].strip().lstrip("-").strip()),
                int(lines[idx + 3].strip().lstrip("-").strip()),
            )
        if line.strip() == "character_dict:":
            for character_line in lines[idx + 1 :]:
                stripped = character_line.strip()
                if not stripped.startswith("- "):
                    break
                characters.append(parse_yaml_scalar(stripped[2:].strip()))
            break

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


def parse_yaml_scalar(value: str) -> str:
    if value == "''''":
        return "'"
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        return ast.literal_eval(value)
    return value
