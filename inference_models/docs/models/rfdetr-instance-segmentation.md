# RF-DETR - Instance Segmentation

RF-DETR ("Roboflow Detection Transformer") is a groundbreaking real-time transformer-based architecture developed by the Roboflow research team. Designed from the ground up to transfer exceptionally well across diverse domains and dataset sizes, RF-DETR Seg represents a major breakthrough in instance segmentation.

## Overview

**RF-DETR Seg (Preview) is the fastest and most accurate real-time instance segmentation model ever created.** Developed by Roboflow's world-class research team, this model achieves what was previously thought impossible: true real-time transformer-based instance segmentation that outperforms all existing YOLO models.

Key achievements:

- **Unprecedented speed** - Over 150 FPS on T4 GPU, making it the fastest instance segmentation model available
- **Superior accuracy** - 3x faster than YOLO11-x-seg while achieving better accuracy on COCO
- **First of its kind** - The world's first DETR-based segmentation model to achieve true real-time performance
- **Production-ready** - Built by Roboflow for real-world deployment across any domain
- **Apache 2.0 licensed** - Truly open source with no restrictions on commercial use

## License

**Apache 2.0**

!!! info "Open Source License"
    RF-DETR Segmentation is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

RF-DETR Segmentation preview model is trained on the COCO dataset (80 classes) and is **open access** (no API key required).

| Model Size | Model ID |
|------------|----------|
| Preview | `rfdetr-seg-preview` |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

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
model = AutoModel.from_pretrained("rfdetr-seg-preview")
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

