# YOLOv11 - Instance Segmentation

YOLOv11 is the latest iteration in the YOLO series developed by Ultralytics. The instance segmentation variant provides state-of-the-art performance with pixel-precise masks for each detected object.

## Overview

YOLOv11 for instance segmentation represents the cutting edge of real-time instance segmentation. Key features include:

- **Enhanced architecture** - Improved backbone and neck design for better feature extraction
- **Pixel-precise masks** - High-quality segmentation masks for each detected object
- **Improved accuracy** - Better performance than YOLOv8 on COCO dataset
- **Efficient inference** - Optimized for real-time applications
- **Multiple model sizes** - From nano to extra-large variants

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLOv11 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLOv11 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLOv11 License Details](https://roboflow.com/model-licenses/yolo11)

## Pre-trained Model IDs

All pre-trained YOLOv11 instance segmentation models are trained on the COCO dataset (80 classes) and are **open access** (no API key required).

| Model Size | 640×640 |
|------------|---------|
| Nano | `yolov11n-seg-640` |
| Small | `yolov11s-seg-640` |
| Medium | `yolov11m-seg-640` |
| Large | `yolov11l-seg-640` |
| Extra-Large | `yolov11x-seg-640` |

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
model = AutoModel.from_pretrained("yolov11n-seg-640")
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

