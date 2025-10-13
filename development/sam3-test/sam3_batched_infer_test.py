import argparse
import os
from time import perf_counter
from typing import List

import numpy as np
from PIL import Image

from inference.core.entities.requests.inference import InferenceRequestImage
from inference.core.entities.requests.sam3 import Sam3Prompt, Sam3SegmentationRequest
from inference.models.sam3.segment_anything3 import SegmentAnything3


def run(image_path: str, prompts: List[str], fmt: str) -> None:
    model = SegmentAnything3()

    # Load image as numpy RGB and pass via InferenceRequestImage(type="numpy")
    np_img = np.array(Image.open(image_path).convert("RGB"))
    req = Sam3SegmentationRequest(
        image=InferenceRequestImage(type="file", value=image_path),
        format=fmt,
        prompts=[Sam3Prompt(type="text", text=p) for p in prompts],
    )

    t0 = perf_counter()
    result = model.infer_from_request(req)
    dt = perf_counter() - t0

    # Print concise summary
    if hasattr(result, "prompt_results"):
        print(f"BATCH response in {dt:.3f}s; prompts={len(result.prompt_results)}")
        for pr in result.prompt_results:
            print(
                f"  idx={pr.prompt_index} type={pr.echo.type} text={pr.echo.text} preds={len(pr.predictions)}"
            )
    elif hasattr(result, "predictions"):
        print(f"SINGLE response in {dt:.3f}s; preds={len(result.predictions)}")
    elif isinstance(result, (bytes, bytearray)):
        out = os.path.join(os.path.dirname(image_path), "sam3_masks.npz")
        with open(out, "wb") as f:
            f.write(result)
        print(f"Binary masks saved to {out} in {dt:.3f}s")
    else:
        print(f"Unknown response type: {type(result)} in {dt:.3f}s")


def parse_args():
    p = argparse.ArgumentParser(description="Test SAM3 batched prompts locally")
    default_img = os.path.join(
        os.path.dirname(__file__),
        "test_image.jpg",
    )
    p.add_argument("--image", type=str, default=default_img)
    p.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default=["cat", "laptop", "keyboard"],
        help="Text prompts to batch (order preserved)",
    )
    p.add_argument(
        "--format",
        type=str,
        default="polygon",
        choices=["polygon", "json", "rle", "binary"],
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(image_path=args.image, prompts=args.prompts, fmt=args.format)
