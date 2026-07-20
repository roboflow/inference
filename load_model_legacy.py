# /// script
# requires-python = ">=3.9,<3.13"
# dependencies = [
#     "inference<1.0",
# ]
# ///
"""Load the models in MODEL_IDS via pre-1.0 inference.get_model and test them.

Example:
    export ROBOFLOW_API_KEY="your-key-here"
    uv run load_model_legacy.py

    uv run load_model_legacy.py --api-key "your-key-here"
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

# This script lives in the inference repo root, so Python would otherwise
# import the repo's current 1.x source tree instead of the installed 0.x
# package. Drop the repo root (and cwd, when equal) from sys.path first.
_REPO_ROOT = Path(__file__).parent.resolve()
sys.path = [
    entry for entry in sys.path if Path(entry if entry else ".").resolve() != _REPO_ROOT
]

# Pre-1.0 inference reads API_BASE_URL, not the newer ROBOFLOW_API_HOST.
# Must be set before the import below, since env.py reads it at import time.
if os.getenv("ROBOFLOW_API_HOST") and not os.getenv("API_BASE_URL"):
    os.environ["API_BASE_URL"] = os.environ["ROBOFLOW_API_HOST"]

# localapi's TLS cert is signed by the local mkcert CA, which certifi does
# not include. Extend (not replace) the bundle so public hosts such as the
# weight-download buckets still verify too.
_MKCERT_ROOT_CA = (
    Path.home() / "Library" / "Application Support" / "mkcert" / "rootCA.pem"
)
if not os.getenv("REQUESTS_CA_BUNDLE") and _MKCERT_ROOT_CA.exists():
    import tempfile

    import certifi

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".pem", prefix="ca-bundle-", delete=False
    ) as _bundle:
        _bundle.write(Path(certifi.where()).read_text())
        _bundle.write("\n")
        _bundle.write(_MKCERT_ROOT_CA.read_text())
    os.environ["REQUESTS_CA_BUNDLE"] = _bundle.name

import numpy as np

from inference import get_model

MODEL_IDS = [
    "the-paid-gallery/182",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Roboflow models with pre-1.0 inference.get_model."
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("ROBOFLOW_API_KEY"),
        help="Roboflow API key (defaults to ROBOFLOW_API_KEY env var).",
    )
    return parser.parse_args()


def load_and_test(model_id: str, api_key: str) -> None:
    print(f"Loading model {model_id!r}...")
    model = get_model(model_id, api_key=api_key)
    print(f"Loaded: {type(model).__name__}")

    print("Running test inference on a random 640x640 image...")
    image = np.random.randint(0, 256, size=(640, 640, 3), dtype=np.uint8)
    responses = model.infer(image)
    if not isinstance(responses, list):
        responses = [responses]
    for response in responses:
        print(f"Response: {response}")
        if hasattr(response, "top"):
            print(
                f"Top class: {response.top!r} "
                f"(confidence={float(response.confidence):.4f})"
            )


def main() -> int:
    args = parse_args()
    if not args.api_key:
        print(
            "Missing API key. Set ROBOFLOW_API_KEY or pass --api-key.",
            file=sys.stderr,
        )
        return 1

    # Pre-1.0 inference caches weights under MODEL_CACHE_DIR (/tmp/cache by
    # default); clear it so every run re-downloads from scratch.
    cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "/tmp/cache"))
    if cache_dir.exists():
        print(f"Clearing model cache at {cache_dir}...")
        shutil.rmtree(cache_dir, ignore_errors=True)

    results = {}
    for index, model_id in enumerate(MODEL_IDS):
        if index > 0:
            print()
        try:
            load_and_test(model_id, api_key=args.api_key)
            results[model_id] = None
        except Exception as error:
            print(f"FAILED {model_id!r}: {error}", file=sys.stderr)
            results[model_id] = error

    succeeded_count = sum(1 for error in results.values() if error is None)
    print(f"\n=== Summary: {succeeded_count}/{len(MODEL_IDS)} model(s) succeeded ===")
    for model_id in MODEL_IDS:
        error = results[model_id]
        if error is None:
            print(f"  OK      {model_id}")
        else:
            print(f"  FAILED  {model_id} ({type(error).__name__}: {error})")

    if succeeded_count < len(MODEL_IDS):
        return 1
    else:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
