# RF-DETR - Semantic Segmentation

RF-DETR Semantic Segmentation assigns a class label to every pixel in an image. It is trained on the Roboflow platform from the RF-DETR instance segmentation weights.

## Overview

The model keeps the RF-DETR encoder and the spatial mask features of the RF-DETR instance segmentation head, and replaces object queries with a per-pixel class projection. Key features include:

- **Per-pixel classification** - Every pixel is assigned a single class label.
- **Background is a class** - Class `0` is the dataset's `background` class, trained like every other class.
- **Sigmoid confidence** - Pixel confidence is the top per-class sigmoid, the score the model is trained and evaluated with, so recommended thresholds keep their meaning.
- **Multiple model sizes** - Nano, Small, Medium, Large, XLarge and 2XLarge, with the geometry of the matching RF-DETR instance segmentation size.

## License

**Apache 2.0**

!!! info "Open Source License"
    RF-DETR Semantic Segmentation is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

There are no public pre-trained semantic checkpoints. Train a model on the Roboflow platform.

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | ❌ |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "my-project-abc123/2",
    api_key="your_roboflow_api_key",
)
image = cv2.imread("path/to/image.jpg")

results = model(image)

seg_map = results[0].segmentation_map      # (H x W) class id per pixel
confidence = results[0].confidence         # (H x W) per-pixel confidence
```

## Output Format

The model returns a list of `SemanticSegmentationResult` objects with:

| Field | Type | Description |
|-------|------|-------------|
| `segmentation_map` | `torch.Tensor` | Class ID for each pixel (H x W) |
| `confidence` | `torch.Tensor` | Confidence score for each pixel (H x W) |
| `image_metadata` | `dict` | Optional metadata about the image |

Pixels with a confidence below the threshold are assigned to the `background` class with confidence `0`.
