"""Compute the reference embeddings the pretrained image-embedding integration tests use.

Every reference comes from an implementation independent of the ONNX packages,
run in float32 on CPU, with weights loaded from `.safetensors` at a pinned revision:

- dinov2/small: transformers `AutoModel` for facebook/dinov2-small with the
  repository's `AutoImageProcessor`. The vector is `pooler_output`, the layer-normed
  CLS token; the script checks it equals `last_hidden_state[:, 0]`.
- siglip2/base-patch16-224: transformers `SiglipVisionModel` (vision tower only)
  for google/siglip2-base-patch16-224 with the repository's `AutoImageProcessor`.
  The vector is `pooler_output`.
- dinov3/vits16: timm `vit_small_patch16_dinov3.lvd1689m`, because the transformers
  repository is gated. The vector is `forward_features(x)[:, 0]`: `forward_features`
  ends with the model's final norm, so this is the layer-normed CLS token. timm's
  evaluation transform runs at the package's preprocessing (224x224 bilinear resize,
  no crop, ImageNet mean and std) instead of timm's default (256, bicubic, center
  crop), which sees a different crop of the image.

Images are decoded with OpenCV, as the tests do, and converted to RGB.

    python -I compute_pretrained_image_embedding_references.py --download-dir /tmp/references
"""

import argparse
import hashlib
import json
import os
import re
from typing import Dict, List

import cv2
import numpy as np
import requests
import timm
import torch
import transformers
from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, SiglipVisionModel

DEFAULT_OUTPUT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tests",
        "integration_tests",
        "models",
        "fixtures",
        "pretrained_image_embeddings_references.json",
    )
)
IMAGE_URLS = {
    "dog": "https://storage.googleapis.com/roboflow-tests-assets/test-images/dog.jpeg",
    "bike": "https://media.roboflow.com/inference/example-input-images/bike.jpg",
    "flowers": "https://media.roboflow.com/inference/example-input-images/flowers.jpg",
}
DINOV2_REPOSITORY = "facebook/dinov2-small"
DINOV2_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
SIGLIP2_REPOSITORY = "google/siglip2-base-patch16-224"
SIGLIP2_REVISION = "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"
DINOV3_TIMM_MODEL = "vit_small_patch16_dinov3.lvd1689m"
DINOV3_REPOSITORY = f"timm/{DINOV3_TIMM_MODEL}"
DINOV3_REVISION = "3bf4720a82ec2066db88137180ff1f83a675cef0"
DINOV3_PACKAGE_TRANSFORM = {
    "input_size": (3, 224, 224),
    "interpolation": "bilinear",
    "crop_pct": 1.0,
    "crop_mode": "squash",
}


def main() -> int:
    args = _parse_args()
    torch.set_grad_enabled(False)
    images = _download_images(os.path.join(args.download_dir, "images"))
    rgb_images = [image["rgb"] for image in images.values()]
    models = {
        "dinov2/small": _dinov2_reference(
            rgb_images, os.path.join(args.download_dir, "dinov2")
        ),
        "dinov3/vits16": _dinov3_reference(
            rgb_images, os.path.join(args.download_dir, "dinov3")
        ),
        "siglip2/base-patch16-224": _siglip2_reference(
            rgb_images, os.path.join(args.download_dir, "siglip2")
        ),
    }
    for model in models.values():
        vectors = model.pop("vectors")
        model["embeddings"] = {
            name: [float(f"{value:.7g}") for value in vector]
            for name, vector in zip(images, vectors.tolist())
        }
    references = {
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "timm": timm.__version__,
        },
        "images": {
            name: {"url": image["url"], "sha256": image["sha256"]}
            for name, image in images.items()
        },
        "models": models,
    }
    _write_references(args.output, references)
    print(f"wrote {args.output}")
    return 0


def _dinov2_reference(images: List[np.ndarray], cache_dir: str) -> dict:
    model = AutoModel.from_pretrained(
        DINOV2_REPOSITORY,
        revision=DINOV2_REVISION,
        use_safetensors=True,
        dtype=torch.float32,
        cache_dir=cache_dir,
    ).eval()
    processor = AutoImageProcessor.from_pretrained(
        DINOV2_REPOSITORY, revision=DINOV2_REVISION, cache_dir=cache_dir
    )
    outputs = model(**processor(images=_to_pil(images), return_tensors="pt"))
    if not torch.equal(outputs.pooler_output, outputs.last_hidden_state[:, 0]):
        raise SystemExit("DINOv2 pooler_output is not the layer-normed CLS token")
    return {
        "implementation": f"transformers {type(model).__name__}",
        "repository": DINOV2_REPOSITORY,
        "revision": DINOV2_REVISION,
        "preprocessing": type(processor).__name__,
        "vector": "pooler_output, equal to last_hidden_state[:, 0]",
        "vectors": outputs.pooler_output,
    }


def _siglip2_reference(images: List[np.ndarray], cache_dir: str) -> dict:
    model = SiglipVisionModel.from_pretrained(
        SIGLIP2_REPOSITORY,
        revision=SIGLIP2_REVISION,
        use_safetensors=True,
        dtype=torch.float32,
        cache_dir=cache_dir,
    ).eval()
    processor = AutoImageProcessor.from_pretrained(
        SIGLIP2_REPOSITORY, revision=SIGLIP2_REVISION, cache_dir=cache_dir
    )
    outputs = model(**processor(images=_to_pil(images), return_tensors="pt"))
    return {
        "implementation": f"transformers {type(model).__name__}",
        "repository": SIGLIP2_REPOSITORY,
        "revision": SIGLIP2_REVISION,
        "preprocessing": type(processor).__name__,
        "vector": "pooler_output",
        "vectors": outputs.pooler_output,
    }


def _dinov3_reference(images: List[np.ndarray], cache_dir: str) -> dict:
    weights = hf_hub_download(
        DINOV3_REPOSITORY,
        "model.safetensors",
        revision=DINOV3_REVISION,
        cache_dir=cache_dir,
    )
    model = timm.create_model(
        DINOV3_TIMM_MODEL,
        pretrained=True,
        pretrained_cfg_overlay={"file": weights},
        img_size=DINOV3_PACKAGE_TRANSFORM["input_size"][1:],
    ).eval()
    data_config = {
        **timm.data.resolve_model_data_config(model),
        **DINOV3_PACKAGE_TRANSFORM,
    }
    transform = timm.data.create_transform(**data_config, is_training=False)
    batch = torch.stack([transform(image) for image in _to_pil(images)])
    vectors = model.forward_features(batch)[:, 0]
    return {
        "implementation": f"timm {DINOV3_TIMM_MODEL}",
        "repository": DINOV3_REPOSITORY,
        "revision": DINOV3_REVISION,
        "preprocessing": f"timm.data.create_transform({_describe(data_config)})",
        "vector": "forward_features(x)[:, 0], the CLS token after the final norm",
        "vectors": vectors,
    }


def _download_images(directory: str) -> Dict[str, dict]:
    os.makedirs(directory, exist_ok=True)
    images = {}
    for name, url in IMAGE_URLS.items():
        path = os.path.join(directory, os.path.basename(url))
        if not os.path.exists(path):
            response = requests.get(url, timeout=60)
            response.raise_for_status()
            with open(path, "wb") as file:
                file.write(response.content)
        with open(path, "rb") as file:
            digest = hashlib.sha256(file.read()).hexdigest()
        image = cv2.imread(path)
        if image is None:
            raise SystemExit(f"could not decode {path}")
        images[name] = {
            "url": url,
            "sha256": digest,
            "rgb": np.ascontiguousarray(image[:, :, ::-1]),
        }
    return images


def _to_pil(images: List[np.ndarray]) -> List[Image.Image]:
    return [Image.fromarray(image) for image in images]


def _describe(config: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in sorted(config.items()))


def _write_references(path: str, references: dict) -> None:
    content = json.dumps(references, indent=2)
    # One line per vector keeps the file short and its diffs readable.
    content = re.sub(
        r"\[\s+(-?[\d.e+-]+(?:,\s+-?[\d.e+-]+)*)\s+\]",
        lambda match: "[" + re.sub(r",\s+", ", ", match.group(1)) + "]",
        content,
    )
    with open(path, "w") as file:
        file.write(content + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--download-dir",
        required=True,
        help="Directory for the images and reference weights, one subdirectory each.",
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
