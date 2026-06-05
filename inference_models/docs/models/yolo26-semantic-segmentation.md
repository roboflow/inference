# YOLO26 - Semantic Segmentation

YOLO26 is the latest addition to the Ultralytics YOLO model series. The semantic segmentation variant assigns a class label to every pixel in an image, producing dense scene-level masks rather than per-object instances.

## Overview

YOLO26 for semantic segmentation pairs the efficient YOLO26 backbone with a dense per-pixel prediction head. Key features include:

- **Per-pixel classification** - Every pixel is assigned a single class label.
- **Efficient YOLO26 backbone** - Shares the NMS-free, DFL-free YOLO26 architecture for fast inference and broad edge compatibility.
- **Cityscapes pre-trained checkpoints** - Public weights trained on the 19-class Cityscapes dataset, available across all model sizes.
- **Binary and multi-class** - Supports a single foreground class (binary head) or many foreground classes, alongside an implicit background.
- **Multiple model sizes** - From nano to extra-large variants.

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLO26 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLO26 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLO26 License Details](https://roboflow.com/model-licenses/yolo26)

## Pre-trained Model IDs

Public YOLO26 semantic segmentation checkpoints are trained on the **Cityscapes** dataset (19 classes) at 1024×1024 and are **open access** (no API key required).

| Model Size | 1024×1024 |
|------------|-----------|
| Nano | `yolo26n-sem-1024` |
| Small | `yolo26s-sem-1024` |
| Medium | `yolo26m-sem-1024` |
| Large | `yolo26l-sem-1024` |
| Extra-Large | `yolo26x-sem-1024` |

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `torch-script` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |
| `trt` | `trt10` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | ✅ Upload pre-trained weights |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

```python
import cv2
import numpy as np
from inference_models import AutoModel

# Load a public Cityscapes checkpoint (open access, no API key required)
model = AutoModel.from_pretrained("yolo26n-sem-1024")
image = cv2.imread("path/to/image.jpg")

# Run inference
results = model(image)

# Per-pixel class IDs and confidence
seg_map = results[0].segmentation_map      # (H x W) class id per pixel
confidence = results[0].confidence         # (H x W) per-pixel confidence

# Colour the mask and overlay on the original image
colors = np.random.randint(0, 255, size=(len(model.class_names), 3), dtype=np.uint8)
colored_mask = colors[seg_map.cpu().numpy()]
overlay = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)
cv2.imwrite("segmentation_result.jpg", overlay)
```

To run a custom-trained model, pass your `project-url/version` and a Roboflow API key:

```python
model = AutoModel.from_pretrained(
    "my-project-abc123/2",
    api_key="your_roboflow_api_key",
)
```

## Output Format

The model returns a list of `SemanticSegmentationResult` objects with:

| Field | Type | Description |
|-------|------|-------------|
| `segmentation_map` | `torch.Tensor` | Class ID for each pixel (H x W) |
| `confidence` | `torch.Tensor` | Confidence score for each pixel (H x W) |
| `image_metadata` | `dict` | Optional metadata about the image |

!!! note "Semantic vs Instance Segmentation"
    Semantic segmentation labels every pixel with a class but does not separate individual objects (all cars share the "car" label). For per-object masks, counting, or tracking, use an instance segmentation model such as [YOLO26 - Instance Segmentation](yolo26-instance-segmentation.md).
