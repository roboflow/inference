#!/usr/bin/env python3
"""Load the models in MODEL_IDS via inference_models.AutoModel and test them.

Example:
    export ROBOFLOW_API_KEY="your-key-here"
    python load_model.py

    python load_model.py --api-key "your-key-here"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np

from inference_models import AutoModel

MODEL_IDS = [
    "kais-workspace-2skmm/danger-noodles-non-square-4-sam3-large-t1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Roboflow models with inference_models.AutoModel."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (defaults to ROBOFLOW_API_KEY env var).",
    )
    return parser.parse_args()


def load_and_test(model_id: str, api_key: str) -> None:
    print(f"Loading model {model_id!r}...")
    model = AutoModel.from_pretrained(model_id, api_key=api_key)
    print(f"Loaded: {type(model).__name__}")
    print(f"Model: {model}")

    print("Running test inference on a random 640x640 image...")
    image = np.random.randint(0, 256, size=(640, 640, 3), dtype=np.uint8)
    prediction = model.infer(image)
    print(f"Prediction: {prediction}")
    if hasattr(model, "class_names") and hasattr(prediction, "class_id"):
        class_id = int(prediction.class_id[0])
        confidence = float(prediction.confidence[0, class_id])
        print(
            f"Top class: {model.class_names[class_id]!r} "
            f"(id={class_id}, confidence={confidence:.4f})"
        )


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print(
            "Missing API key. Set ROBOFLOW_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/cache"))
    if cache_dir.exists():
        print(f"Clearing model cache at {cache_dir}...")
        shutil.rmtree(cache_dir, ignore_errors=True)

    failures = []
    for index, model_id in enumerate(MODEL_IDS):
        if index > 0:
            print()
        try:
            load_and_test(model_id, api_key=args.api_key)
        except Exception as error:
            print(f"FAILED {model_id!r}: {error}", file=sys.stderr)
            failures.append(model_id)

    if failures:
        print(
            f"\n{len(failures)}/{len(MODEL_IDS)} model(s) failed: {failures}",
            file=sys.stderr,
        )
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
