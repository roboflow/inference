import argparse
import base64
import json
import time
from io import BytesIO
from typing import Dict, Tuple

import requests
from PIL import Image


def _encode_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def _load_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def bench_json_old(
    server: str, model_id: str, image_path: str, api_key: str, iters: int
) -> Tuple[float, Dict]:
    url = f"{server}/infer/object_detection"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    img_b64 = _encode_image_base64(image_path)
    payload = {
        "model_id": model_id,
        "image": {"type": "base64", "value": img_b64},
        "confidence": 0.4,
        "iou_threshold": 0.3,
    }
    # Warmup
    requests.post(url, headers=headers, json=payload, timeout=60)
    t0 = time.perf_counter()
    last = None
    for _ in range(iters):
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        last = resp.json()
    t1 = time.perf_counter()
    return (t1 - t0) / iters, last


def bench_multipart_v1(
    server: str, model_id: str, image_path: str, api_key: str, iters: int
) -> Tuple[float, Dict]:
    url = f"{server}/v1/infer/object_detection"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    # Warmup
    with open(image_path, "rb") as f:
        requests.post(
            url,
            headers=headers,
            data={"model_id": model_id},
            files={"image": ("image.jpg", f, "image/jpeg")},
            timeout=60,
        )
    t0 = time.perf_counter()
    last = None
    for _ in range(iters):
        with open(image_path, "rb") as f:
            resp = requests.post(
                url,
                headers=headers,
                data={"model_id": model_id},
                files={"image": ("image.jpg", f, "image/jpeg")},
                timeout=60,
            )
        resp.raise_for_status()
        last = resp.json()
    t1 = time.perf_counter()
    return (t1 - t0) / iters, last


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark JSON+base64 vs multipart /v1 for object detection."
    )
    parser.add_argument("--server", default="http://localhost:9001")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--api-key", default="")
    parser.add_argument("--iters", type=int, default=10)
    args = parser.parse_args()

    json_avg, json_resp = bench_json_old(
        args.server, args.model_id, args.image, args.api_key, args.iters
    )
    mp_avg, mp_resp = bench_multipart_v1(
        args.server, args.model_id, args.image, args.api_key, args.iters
    )

    def _summarize(r: Dict) -> Tuple[int, int]:
        # basic shape: predictions length and first prediction keys
        if isinstance(r, list):
            r0 = r[0]
        else:
            r0 = r
        preds = r0.get("predictions", [])
        return len(preds), len(r0.keys())

    json_preds, json_keys = _summarize(json_resp)
    mp_preds, mp_keys = _summarize(mp_resp)
    print("Results:")
    print(f"- JSON+base64 avg latency: {json_avg*1000:.2f} ms, preds={json_preds}, keys={json_keys}")
    print(f"- multipart /v1 avg latency: {mp_avg*1000:.2f} ms, preds={mp_preds}, keys={mp_keys}")


if __name__ == "__main__":
    main()


