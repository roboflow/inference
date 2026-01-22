# YOLOv11 - Keypoint Detection

YOLOv11 is the latest evolution in the YOLO family developed by Ultralytics. The keypoint detection variant (also known as pose estimation) detects objects and predicts keypoints for each detected instance, commonly used for human pose estimation.

## Overview

YOLOv11 for keypoint detection combines object detection with keypoint localization. Key features include:

- **Improved architecture** - Enhanced backbone and neck design over YOLOv8
- **Keypoint prediction** - Predicts x, y coordinates and confidence for each keypoint
- **Multi-person support** - Detects and tracks keypoints for multiple people simultaneously
- **Better accuracy** - Improved performance on pose estimation benchmarks
- **Multiple model sizes** - From nano to extra-large variants

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLOv11 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLOv11 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLOv11 License Details](https://roboflow.com/model-licenses/yolo11)

## Pre-trained Model IDs

YOLOv11 keypoint detection models support **custom models only** (no pre-trained COCO models available). Train your own keypoint detection model on Roboflow and deploy it using the model ID format below.

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
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Keypoint Detection block |
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

# Run inference - returns (List[KeyPoints], Optional[List[Detections]])
results = model(image)
key_points_list, detections_list = results

# Convert to supervision format for filtering and visualization
key_points = key_points_list[0].to_supervision()

# Filter for specific class (e.g., class_id=0)
class_mask = key_points.class_id == 0
key_points_filtered = key_points[class_mask]

# Annotate image with keypoints
vertex_annotator = sv.VertexAnnotator()
edge_annotator = sv.EdgeAnnotator(edges=model.skeletons[0])

annotated_image = edge_annotator.annotate(image.copy(), key_points_filtered)
annotated_image = vertex_annotator.annotate(annotated_image, key_points_filtered)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```

## Output Format

Returns: `Tuple[List[KeyPoints], Optional[List[Detections]]]` (one element per image in batch)

**Skeleton connections**: Access via `model.skeletons[class_id]`. The `model.skeletons` list has one element per detection class. For single-class models, use `model.skeletons[0]`. For multi-class models, each class has its own skeleton at the corresponding index.

Pass to `EdgeAnnotator`: `sv.EdgeAnnotator(edges=model.skeletons[0])` or omit for auto-detection.

