"""Benchmark sync HTTP session reuse for hosted inference requests.

This script compares the old effective behavior, where each timed request uses
a fresh HTTP session, against the SDK executor's thread-local session reuse.

Example:
    ROBOFLOW_API_KEY=... python development/benchmark_scripts/benchmark_remote_http_session_reuse.py \
        --model-id coco/3 \
        --iterations 100
"""

import argparse
import json
import math
import os
import statistics
import time
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit, urlunsplit

import requests
from PIL import Image

from inference_sdk import InferenceConfiguration, InferenceHTTPClient
from inference_sdk.http.utils import executors

DEFAULT_API_URL = "https://serverless.roboflow.com"
DEFAULT_IMAGE_URL = "https://media.roboflow.com/inference/seawithdock.jpeg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare fresh HTTP sessions with thread-local session reuse for "
            "sync inference_sdk requests."
        )
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("ROBOFLOW_API_URL", DEFAULT_API_URL),
        help=f"Inference API URL. Defaults to {DEFAULT_API_URL}.",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("ROBOFLOW_MODEL_ID", "coco/3"),
        help="Model id to benchmark. Defaults to ROBOFLOW_MODEL_ID or coco/3.",
    )
    parser.add_argument(
        "--image",
        default=os.getenv("ROBOFLOW_BENCHMARK_IMAGE", DEFAULT_IMAGE_URL),
        help="Image path or URL. URL inputs are downloaded once before timing.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=50,
        help="Timed requests per pass.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Untimed warmup requests per pass.",
    )
    return parser.parse_args()


def load_image(image_reference: str) -> Image.Image:
    if image_reference.startswith(("http://", "https://")):
        response = requests.get(image_reference, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    return Image.open(Path(image_reference)).convert("RGB")


def build_client(api_url: str, api_key: str) -> InferenceHTTPClient:
    client = InferenceHTTPClient(api_url=api_url.rstrip("/"), api_key=api_key)
    parsed_url = urlsplit(api_url if "://" in api_url else f"//{api_url}")
    hostname = parsed_url.hostname or ""
    if hostname == "roboflow.com" or hostname.endswith(".roboflow.com"):
        client.select_api_v0()
    return client.configure(
        inference_configuration=InferenceConfiguration(
            disable_active_learning=True,
            max_batch_size=1,
            max_concurrent_requests=1,
            source="remote-http-session-reuse-benchmark",
        )
    )


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    sorted_values = sorted(values)
    index = math.ceil((p / 100) * len(sorted_values)) - 1
    index = min(max(index, 0), len(sorted_values) - 1)
    return sorted_values[index]


def summarize(durations: List[float]) -> Dict[str, Any]:
    total = sum(durations)
    return {
        "requests": len(durations),
        "total_s": total,
        "fps": len(durations) / total if total else None,
        "avg_ms": statistics.fmean(durations) * 1000 if durations else None,
        "median_ms": statistics.median(durations) * 1000 if durations else None,
        "min_ms": min(durations) * 1000 if durations else None,
        "max_ms": max(durations) * 1000 if durations else None,
        "p90_ms": percentile(durations, 90) * 1000 if durations else None,
        "p95_ms": percentile(durations, 95) * 1000 if durations else None,
    }


def run_pass(
    *,
    client: InferenceHTTPClient,
    image: Image.Image,
    model_id: str,
    iterations: int,
    warmup: int,
    reset_session_each_request: bool,
) -> Dict[str, Any]:
    for _ in range(warmup):
        if reset_session_each_request:
            executors._reset_thread_local_requests_session()
        client.infer(inference_input=image, model_id=model_id)

    durations = []
    for _ in range(iterations):
        if reset_session_each_request:
            executors._reset_thread_local_requests_session()
        start = time.perf_counter()
        client.infer(inference_input=image, model_id=model_id)
        durations.append(time.perf_counter() - start)

    executors._reset_thread_local_requests_session()
    return summarize(durations=durations)


def redact_source(value: str) -> str:
    if not value.startswith(("http://", "https://")):
        return value
    parts = urlsplit(value)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))


def main() -> None:
    args = parse_args()
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY must be set.")

    image = load_image(image_reference=args.image)
    client = build_client(api_url=args.api_url, api_key=api_key)

    fresh_session = run_pass(
        client=client,
        image=image,
        model_id=args.model_id,
        iterations=args.iterations,
        warmup=args.warmup,
        reset_session_each_request=True,
    )
    reused_session = run_pass(
        client=client,
        image=image,
        model_id=args.model_id,
        iterations=args.iterations,
        warmup=args.warmup,
        reset_session_each_request=False,
    )

    result = {
        "api_url": args.api_url,
        "model_id": args.model_id,
        "image": {
            "source": redact_source(args.image),
            "width": image.width,
            "height": image.height,
        },
        "iterations": args.iterations,
        "warmup": args.warmup,
        "modes": {
            "fresh_session_per_request": fresh_session,
            "thread_local_session_reuse": reused_session,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
