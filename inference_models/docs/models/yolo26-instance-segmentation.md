# YOLO26 - Instance Segmentation

YOLO26 is the latest addition to the Ultralytics YOLO model series. The instance segmentation variant extends object detection capabilities by providing pixel-precise masks for each detected object.

## Overview

YOLO26 for instance segmentation combines NMS-free object detection with pixel-level segmentation masks. Key features include:

- **NMS-free end-to-end inference** - Removing non-maximum suppression helps achieve lower inference latencies.
- **DFL removal** - Distribution Focal Loss removed for simpler export and broader edge compatibility
- **Semantic segmentation loss** - Improves model convergence
- **Upgraded proto module** - Uses multi-scale information to produce higher-quality masks
- **Multiple model sizes** - From nano to extra-large variants

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLO26 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLO26 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLO26 License Details](https://roboflow.com/model-licenses/yolo26)

## Pre-trained Model IDs

All pre-trained YOLO26 instance segmentation models are trained on the COCO dataset (80 classes) and are **open access** (no API key required).

| Model Size | 640×640 |
|------------|---------|
| Nano | `yolo26n-seg` |
| Small | `yolo26s-seg` |
| Medium | `yolo26m-seg` |
| Large | `yolo26l-seg` |
| Extra-Large | `yolo26x-seg` |

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `torch-script` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | ✅ Upload pre-trained weights |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Instance Segmentation block |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load your custom model (requires Roboflow API key)
model = AutoModel.from_pretrained(
    "my-project-abc123/2",
    api_key="your_roboflow_api_key"
)
image = cv2.imread("path/to/image.jpg")

# Run inference
predictions = model(image)
detections = predictions[0].to_supervision()

# Annotate image with masks
mask_annotator = sv.MaskAnnotator()
annotated_image = mask_annotator.annotate(image.copy(), detections)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```
