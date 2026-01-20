# YOLOv5 - Instance Segmentation

YOLOv5 is a widely-used object detection model developed by Ultralytics. The instance segmentation variant extends the model to provide pixel-level masks for detected objects.

## Overview

YOLOv5 for instance segmentation provides a balance of speed and accuracy for real-time applications. Key features include:

- **Proven architecture** - Battle-tested model used in thousands of production applications
- **Pixel-level masks** - Detailed segmentation for each detected object
- **Fast inference** - Optimized for real-time performance
- **Easy to deploy** - Well-supported across multiple platforms
- **Multiple backends** - ONNX and TensorRT support

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLOv5 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLOv5 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLOv5 License Details](https://roboflow.com/model-licenses/yolov5)

## Pre-trained Model IDs

YOLOv5 instance segmentation models must be trained on Roboflow or uploaded as custom weights. There are no pre-trained public model IDs available.

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `trt` | `trt10` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | ✅ Upload pre-trained weights ([guide](https://docs.roboflow.com/deploy/upload-custom-weights)) |
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

