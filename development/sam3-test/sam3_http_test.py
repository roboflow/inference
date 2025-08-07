#!/usr/bin/env python3
import argparse
import base64
import json
from pathlib import Path

import requests
from PIL import Image


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="SAM3 HTTP segment_image test")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument("--text", type=str, default=None, help="Text prompt for OV segmentation")
    parser.add_argument("--x", type=float, default=None, help="Box x (pixels)")
    parser.add_argument("--y", type=float, default=None, help="Box y (pixels)")
    parser.add_argument("--w", type=float, default=None, help="Box w (pixels)")
    parser.add_argument("--h", type=float, default=None, help="Box h (pixels)")
    parser.add_argument(
        "--url", default="http://127.0.0.1:9001", help="Inference server base URL"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5, help="Output prob threshold"
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    # Prepare request
    img_b64 = encode_image_base64(str(img_path))
    payload = {
        "format": "json",
        "image": {"type": "base64", "value": img_b64},
        "output_prob_thresh": args.threshold,
    }

    if args.text:
        payload["text"] = args.text
    else:
        # Require a box if no text
        if None in (args.x, args.y, args.w, args.h):
            raise SystemExit("When --text is not provided, you must pass --x --y --w --h for a box prompt")
        # Get image size to normalize box
        with Image.open(img_path) as im:
            width, height = im.size
        nx = args.x / width
        ny = args.y / height
        nw = args.w / width
        nh = args.h / height
        payload["boxes"] = [[nx, ny, nw, nh]]
        payload["box_labels"] = [1]

    endpoint = args.url.rstrip("/") + "/sam3/segment_image"
    resp = requests.post(endpoint, json=payload, timeout=120)
    resp.raise_for_status()

    # Pretty-print JSON response
    try:
        print(json.dumps(resp.json(), indent=2))
    except requests.exceptions.JSONDecodeError:
        print("Non-JSON response received (did you request binary format?)")
        print(f"Status: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('Content-Type')}")
        print(f"Body bytes: {len(resp.content)}")


if __name__ == "__main__":
    main()


