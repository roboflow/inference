"""Measure end-to-end cold model loads with and without the shared blob cache.

The cache must already contain the model blobs for the cache-hit run. This script
uses a fresh local INFERENCE_HOME for every measurement and reports observed
wall-clock durations; it does not delete or mutate an existing local model cache.

Example:
    MODEL_BLOB_CACHE_ENABLED=true \
    MODEL_BLOB_CACHE_BUCKET=model-cache \
    MODEL_BLOB_CACHE_ENDPOINT_URL=https://objects.example.com \
    python development/profiling/benchmark_model_blob_cache.py yolov8n-640
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time


def _load_model(model_id: str) -> None:
    from inference_models import AutoModel

    started_at = time.perf_counter()
    AutoModel.from_pretrained(model_id, verbose=False)
    print(json.dumps({"seconds": time.perf_counter() - started_at}))


def _measure(model_id: str, cache_enabled: bool) -> float:
    with tempfile.TemporaryDirectory(prefix="inference-models-benchmark-") as cache_dir:
        environment = os.environ.copy()
        environment["INFERENCE_HOME"] = cache_dir
        environment["MODEL_BLOB_CACHE_ENABLED"] = str(cache_enabled).lower()
        result = subprocess.run(
            [sys.executable, __file__, model_id, "--worker"],
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
    return float(json.loads(result.stdout.strip().splitlines()[-1])["seconds"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_id")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    if arguments.worker:
        _load_model(arguments.model_id)
        return

    without_cache = _measure(arguments.model_id, cache_enabled=False)
    with_cache = _measure(arguments.model_id, cache_enabled=True)
    print(
        json.dumps(
            {
                "model_id": arguments.model_id,
                "cold_load_without_blob_cache_seconds": without_cache,
                "cold_load_with_blob_cache_seconds": with_cache,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
