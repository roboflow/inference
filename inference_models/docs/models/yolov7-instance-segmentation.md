# YOLOv7 - Instance Segmentation

YOLOv7 is an object detection model that introduced several architectural improvements over YOLOv5. The instance segmentation variant provides pixel-level masks for detected objects.

## Overview

YOLOv7 for instance segmentation offers improved accuracy and speed compared to earlier YOLO versions. Key features include:

- **E-ELAN architecture** - Efficient Layer Aggregation Network for better feature extraction
- **Pixel-level masks** - Detailed segmentation for each detected object
- **Improved accuracy** - Better performance than YOLOv5 on many datasets
- **Fast inference** - Optimized for real-time applications
- **Multiple backends** - ONNX and TensorRT support

## License

**AGPL-3.0**

!!! warning "License Notice"
    YOLOv7 is licensed under AGPL-3.0. This is a copyleft license that requires you to open-source any modifications or derivative works, and any software that uses YOLOv7 must also be open-sourced under AGPL-3.0.

    **Roboflow does not provide commercial licensing for YOLOv7.** If you need to use YOLOv7 commercially without open-sourcing your code, you must obtain a license directly from the YOLOv7 authors or consider using a different model architecture.

    For commercial-friendly alternatives, consider:
    - **RF-DETR Seg** (Apache 2.0) - Faster and more accurate
    - **YOLOv8 Seg** (AGPL-3.0, but Roboflow provides commercial licensing)
    - **YOLOv11 Seg** (AGPL-3.0, but Roboflow provides commercial licensing)

## Pre-trained Model IDs

YOLOv7 instance segmentation models must be trained on Roboflow or uploaded as custom weights. There are no pre-trained public model IDs available.

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

