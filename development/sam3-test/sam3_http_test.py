#!/usr/bin/env python3
import argparse
import base64
import json
from pathlib import Path
from typing import List, Tuple

import requests
from PIL import Image, ImageDraw


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def main():
    parser = argparse.ArgumentParser(description="SAM3 HTTP segment_image test")
    parser.add_argument("--image", required=True, help="Path to image file")
    parser.add_argument(
        "--text", type=str, default=None, help="Text prompt for OV segmentation"
    )
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
    parser.add_argument(
        "--model_id",
        type=str,
        default=None,
        help="Model ID to use (e.g., 'sam3' or full model path)",
    )
    parser.add_argument(
        "--out", type=str, default="result.jpg", help="Output path for visualization"
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f"Image not found: {img_path}")

    # Prepare request
    img_b64 = encode_image_base64(str(img_path))
    payload = {
        "format": "rle",
        "image": {"type": "base64", "value": img_b64},
        "output_prob_thresh": args.threshold,
        "api_key": "bYnuUL7O8JxPMe8KM0N0",
    }

    if args.model_id:
        payload["model_id"] = args.model_id

    if args.text:
        payload["text"] = args.text
    else:
        # Require a box if no text
        if None in (args.x, args.y, args.w, args.h):
            raise SystemExit(
                "When --text is not provided, you must pass --x --y --w --h for a box prompt"
            )
        # Get image size to normalize box
        with Image.open(img_path) as im:
            width, height = im.size
        nx = args.x / width
        ny = args.y / height
        nw = args.w / width
        nh = args.h / height
        payload["boxes"] = [[nx, ny, nw, nh]]
        payload["box_labels"] = [1]

    endpoint = args.url.rstrip("/") + "/seg-preview/segment_image"
    resp = requests.post(endpoint, json=payload, timeout=120)
    resp.raise_for_status()

    # Try to parse JSON and visualize
    try:
        data = resp.json()
        print(json.dumps(data, indent=2))
        # Visualize predictions
        preds = data.get("predictions", [])
        if preds:
            with Image.open(img_path).convert("RGBA") as base:
                overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
                draw = ImageDraw.Draw(overlay, "RGBA")

                # Simple palette
                colors = [
                    (72, 146, 234, 96),  # blue
                    (0, 238, 195, 96),  # teal
                    (254, 78, 240, 96),  # magenta
                    (244, 0, 78, 96),  # red
                    (250, 114, 0, 96),  # orange
                    (238, 238, 23, 96),  # yellow
                    (144, 255, 0, 96),  # green
                ]

                def to_xy(points: List[List[float]]) -> List[Tuple[float, float]]:
                    return [(float(x), float(y)) for x, y in points]

                for i, pred in enumerate(preds):
                    color = colors[i % len(colors)]
                    for poly in pred.get("masks", []):
                        if len(poly) >= 3:
                            draw.polygon(
                                to_xy(poly), fill=color, outline=color[:3] + (180,)
                            )

                result = Image.alpha_composite(base, overlay).convert("RGB")
                out_path = Path(args.out)
                result.save(out_path, format="JPEG")
                print(f"Saved visualization to {out_path.resolve()}")
        else:
            print("No predictions to visualize.")
    except requests.exceptions.JSONDecodeError:
        print("Non-JSON response received (did you request binary format?)")
        print(f"Status: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('Content-Type')}")
        print(f"Body bytes: {len(resp.content)}")


if __name__ == "__main__":
    main()
