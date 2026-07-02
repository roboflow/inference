#!/usr/bin/env python3
"""Minimal reproducible example: PP-OCRv6 two-stage OCR (detect -> recognize).

Renders a self-contained multi-line text image, downloads the public PP-OCRv6
ONNX detection and recognition packages from the Hugging Face Hub (no token
required), then runs the full pipeline:

    detect text lines  ->  perspective-crop each line  ->  recognize each crop

and asserts that the transcription matches the rendered text.

Run from the ``inference_models`` project root (sibling of ``examples/``)::

    uv run python examples/pp_ocrv6/two_stage_ocr.py

Or from the monorepo root (imports resolve to the inner package)::

    uv run python inference_models/examples/pp_ocrv6/two_stage_ocr.py

Once the models are registered in the Roboflow model registry, the two
``from_pretrained(...)`` calls below can be replaced with
``AutoModel.from_pretrained("pp-ocrv6-det/medium")`` /
``AutoModel.from_pretrained("pp-ocrv6-rec/small")``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve the inner ``inference_models`` package (…/inference_models/inference_models/)
# when this file is run as a script, so ``PYTHONPATH=./`` at repo root does not shadow
# it with the outer ``inference_models/`` directory name.
_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import cv2
import numpy as np
from huggingface_hub import snapshot_download
from PIL import Image, ImageDraw, ImageFont

from inference_models.models.pp_ocrv6.pp_ocrv6_detection_onnx import (
    PPOCRv6DetectionOnnx,
)
from inference_models.models.pp_ocrv6.pp_ocrv6_recognition_onnx import (
    PPOCRv6RecognitionOnnx,
)

DET_REPO = "PaddlePaddle/PP-OCRv6_medium_det_onnx"
REC_REPO = "PaddlePaddle/PP-OCRv6_small_rec_onnx"
CPU_PROVIDERS = ["CPUExecutionProvider"]
PACKAGE_FILES = ["inference.onnx", "inference.yml"]

TEXT_LINES = [
    "The quick brown fox",
    "jumps over the lazy dog",
    "OCR pipeline test 12345",
]

_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial.ttf",
    "C:\\Windows\\Fonts\\arial.ttf",
]


def _find_font() -> str:
    for candidate in _FONT_CANDIDATES:
        if os.path.exists(candidate):
            return candidate
    try:
        from matplotlib import font_manager

        return font_manager.findfont("DejaVu Sans")
    except Exception as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "No TrueType font found to render the sample image. Install a font "
            "such as DejaVu Sans, or pass one of the paths in _FONT_CANDIDATES."
        ) from error


def render_text_image(lines: list[str]) -> np.ndarray:
    """Render ``lines`` onto a white canvas and return a BGR image."""
    font = ImageFont.truetype(_find_font(), 32)
    image = Image.new("RGB", (640, 60 + 70 * len(lines)), "white")
    draw = ImageDraw.Draw(image)
    for index, line in enumerate(lines):
        draw.text((30, 30 + index * 70), line, fill="black", font=font)
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def rotate_crop(image: np.ndarray, quad: list) -> np.ndarray:
    """Perspective-crop the quadrilateral text region into an upright rectangle."""
    quad = np.array(quad, dtype="float32")
    width = int(
        max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]))
    )
    height = int(
        max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))
    )
    matrix = cv2.getPerspectiveTransform(
        quad, np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    )
    crop = cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
        flags=cv2.INTER_CUBIC,
    )
    if height > 0 and width > 0 and height / float(width) >= 1.5:
        crop = np.rot90(crop)
    return crop


def reading_order(detections) -> list:
    """Group boxes into text lines (top-to-bottom), each ordered left-to-right."""
    items = list(zip(detections.xyxy.tolist(), detections.bboxes_metadata))
    items.sort(key=lambda item: item[0][1])
    lines, current, line_bottom = [], [], None
    for xyxy, meta in items:
        top, bottom = xyxy[1], xyxy[3]
        if line_bottom is None or top >= line_bottom - 0.5 * (bottom - top):
            if current:
                lines.append(current)
            current, line_bottom = [(xyxy, meta)], bottom
        else:
            current.append((xyxy, meta))
            line_bottom = max(line_bottom, bottom)
    if current:
        lines.append(current)
    ordered = []
    for line in lines:
        ordered.extend(sorted(line, key=lambda item: item[0][0]))
    return ordered


def _load(model_cls, repo: str):
    package_dir = snapshot_download(repo, allow_patterns=PACKAGE_FILES)
    return model_cls.from_pretrained(
        package_dir, onnx_execution_providers=CPU_PROVIDERS
    )


def main() -> None:
    print(f"Loading detection model  {DET_REPO!r} …")
    detector = _load(PPOCRv6DetectionOnnx, DET_REPO)
    print(f"Loading recognition model {REC_REPO!r} …")
    recognizer = _load(PPOCRv6RecognitionOnnx, REC_REPO)

    image = render_text_image(TEXT_LINES)

    detections = detector(image)[0]
    ordered = reading_order(detections)
    crops = [rotate_crop(image, meta["polygon"]) for _, meta in ordered]
    texts = recognizer(crops)

    print(f"\nDetected {len(ordered)} text lines:")
    for index, text in enumerate(texts):
        print(f"  [{index}] {text!r}")

    print("\n--- assembled transcription ---")
    print("\n".join(texts))

    assert texts == TEXT_LINES, (
        "Transcription did not match the rendered text.\n"
        f"expected: {TEXT_LINES}\n"
        f"got:      {texts}"
    )
    print("\nOK: transcription matches the rendered text.")


if __name__ == "__main__":
    main()
