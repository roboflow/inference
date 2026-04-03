# YOLOv8 - Instance Segmentation

YOLOv8 is a popular computer vision model developed by Ultralytics. The instance segmentation variant extends object detection capabilities by providing pixel-precise masks for each detected object.

## Overview

YOLOv8 for instance segmentation combines object detection with pixel-level segmentation masks. Key features include:

- **Anchor-free detection head** - Simplified architecture without anchor boxes
- **Pixel-precise masks** - Detailed segmentation for each detected object
- **Improved feature pyramid network** - Better multi-scale feature fusion
- **Efficient backbone** - Optimized for speed and accuracy
- **Multiple model sizes** - From nano to extra-large variants

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLOv8 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLOv8 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLOv8 License Details](https://roboflow.com/model-licenses/yolov8)

## Pre-trained Model IDs

All pre-trained YOLOv8 instance segmentation models are trained on the COCO dataset (80 classes) and are **open access** (no API key required).

| Model Size | 640×640 | 1280×1280 |
|------------|---------|-----------|
| Nano | `yolov8n-seg-640` | `yolov8n-seg-1280` |
| Small | `yolov8s-seg-640` | `yolov8s-seg-1280` |
| Medium | `yolov8m-seg-640` | `yolov8m-seg-1280` |
| Large | `yolov8l-seg-640` | `yolov8l-seg-1280` |
| Extra-Large | `yolov8x-seg-640` | `yolov8x-seg-1280` |

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
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Instance Segmentation block |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model and image
model = AutoModel.from_pretrained("yolov8n-seg-640")
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

