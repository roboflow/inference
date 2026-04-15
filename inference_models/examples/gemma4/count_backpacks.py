#!/usr/bin/env python3
"""End-to-end Gemma 4 example using ``inference_models.AutoModel`` (no CLI arguments).

Downloads a public sample image, loads a hosted Gemma 4 checkpoint via Roboflow, and
asks a focused counting question.

Run from the ``inference_models`` package root::

    uv run python examples/gemma4/run_gemma4_local.py

Optional: set ``GEMMA4_MODEL_ID`` to override the default Roboflow registry id.
"""

from __future__ import annotations

import io
import os
import sys

import numpy as np
import requests
from PIL import Image

from inference_models import AutoModel
from inference_models.configuration import DEFAULT_DEVICE

# Same image used in repo docs (e.g. workflows benchmarks).
IMAGE_URL = "https://media.roboflow.com/inference/people-walking.jpg"

# Roboflow registry id (must match a registered Gemma 4 package).
DEFAULT_MODEL_ID = "gemma-4-e2b-it"

SYSTEM_PROMPT = (
    "You are a precise vision assistant. When asked about people or objects in a scene, "
    "base your answer only on what is clearly visible. If you are uncertain, say so. "
    "For counting questions, give a single best estimate and briefly note any ambiguity "
    "(e.g. partially occluded figures or unclear backpacks)."
)

USER_PROMPT = (
    "How many people in this image are clearly wearing a backpack? "
    "Answer with a number first, then one short sentence explaining what you counted."
)


def _build_prompt(user: str, system: str) -> str:
    return f"{user}<system_prompt>{system}"


def _load_image_rgb(url: str) -> np.ndarray:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return np.array(image)


def main() -> None:
    load_kw = {
        "device": DEFAULT_DEVICE,
        "backend": "hugging-face",
    }
    print(f"Loading hosted model {DEFAULT_MODEL_ID!r} …")
    model = AutoModel.from_pretrained(DEFAULT_MODEL_ID, **load_kw)

    print(f"Fetching image {IMAGE_URL!r} …")
    image_rgb = _load_image_rgb(IMAGE_URL)
    prompt = _build_prompt(USER_PROMPT, SYSTEM_PROMPT)

    print("Running inference …")
    outputs = model.prompt(
        images=image_rgb,
        prompt=prompt,
        input_color_format="rgb",
        max_new_tokens=256,
        do_sample=False,
    )
    print("---")
    print(outputs[0] if outputs else outputs)


if __name__ == "__main__":
    main()
