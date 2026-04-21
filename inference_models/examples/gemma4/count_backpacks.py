#!/usr/bin/env python3
"""Downloads a public sample image, loads a hosted Gemma 4 checkpoint, and asks a focused counting question.

Run from the ``inference_models`` project root (sibling of ``examples/``)::

    uv run python examples/gemma4/count_backpacks.py

Or from the monorepo root (no editable install needed; imports resolve to
``inference_models/inference_models/__init__.py``)::

    uv run --env-file .env inference_models/examples/gemma4/count_backpacks.py
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

# Resolve the inner ``inference_models`` package (…/inference_models/inference_models/)
# when this file is run as a script, so ``PYTHONPATH=./`` at repo root does not shadow
# it with the outer ``inference_models/`` directory name.
_pkg_root = Path(__file__).resolve().parents[2]
if str(_pkg_root) not in sys.path:
    sys.path.insert(0, str(_pkg_root))

import numpy as np
import requests
from PIL import Image

from inference_models import AutoModel
from inference_models.configuration import DEFAULT_DEVICE

# Roboflow registry id (must match a registered Gemma 4 package).
DEFAULT_MODEL_ID = "gemma-4-e2b-it"
IMAGE_URL = "https://media.roboflow.com/inference/people-walking.jpg"

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
