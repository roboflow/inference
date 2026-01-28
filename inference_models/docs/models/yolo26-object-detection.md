# YOLO26 - Object Detection

YOLO26 is the latest addition to the Ultralytics YOLO object detection model series. It achieves superior speed and accuracy compared to prior models in the YOLO series.

## Overview

YOLO26 for object detection removes unnecessary complexity while integrating targeted innovations. Key features include:

- **NMS-free end-to-end inference** - Native end-to-end predictions without non-maximum suppression
- **DFL removal** - Distribution Focal Loss removed for simpler export and broader edge compatibility
- **MuSGD optimizer** - Hybrid SGD/Muon optimizer for more stable training
- **ProgLoss + STAL** - Improved loss functions for better small-object detection
- **Multiple model sizes** - From nano to extra-large variants

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLO26 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLO26 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLO26 License Details](https://roboflow.com/model-licenses/yolo26)

## Pre-trained Model IDs

All pre-trained YOLO26 object detection models are trained on the COCO dataset (80 classes) and are **open access** (no API key required).

| Model Size | 640×640 |
|------------|---------|
| Nano | `yolo26n` |
| Small | `yolo26s` |
| Medium | `yolo26m` |
| Large | `yolo26l` |
| Extra-Large | `yolo26x` |

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
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Object Detection block |
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

# Run inference and convert to supervision Detections
predictions = model(image)
detections = predictions[0].to_supervision()

# Annotate image
bounding_box_annotator = sv.BoxAnnotator()
annotated_image = bounding_box_annotator.annotate(image, detections)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```
