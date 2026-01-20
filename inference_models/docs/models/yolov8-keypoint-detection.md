# YOLOv8 - Keypoint Detection

YOLOv8 is a popular computer vision model developed by Ultralytics. The keypoint detection variant (also known as pose estimation) detects objects and predicts keypoints for each detected instance, commonly used for human pose estimation.

## Overview

YOLOv8 for keypoint detection combines object detection with keypoint localization. Key features include:

- **Anchor-free detection head** - Simplified architecture without anchor boxes
- **Keypoint prediction** - Predicts x, y coordinates and confidence for each keypoint
- **Multi-person support** - Detects and tracks keypoints for multiple people simultaneously
- **Improved feature pyramid network** - Better multi-scale feature fusion
- **Multiple model sizes** - From nano to extra-large variants

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLOv8 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLOv8 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLOv8 License Details](https://roboflow.com/model-licenses/yolov8)

## Pre-trained Model IDs

All pre-trained YOLOv8 keypoint detection models are trained on the COCO dataset (17 keypoints for human pose) and are **open access** (no API key required).

| Model Size | 640×640 | 1280×1280 |
|------------|---------|-----------|
| Nano | `yolov8n-pose-640` | - |
| Small | `yolov8s-pose-640` | - |
| Medium | `yolov8m-pose-640` | - |
| Large | `yolov8l-pose-640` | - |
| Extra-Large | `yolov8x-pose-640` | `yolov8x-pose-1280` |

**COCO Keypoints**: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

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

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model and image
model = AutoModel.from_pretrained("yolov8n-pose-640")
image = cv2.imread("path/to/image.jpg")

# Run inference - returns (List[KeyPoints], Optional[List[Detections]])
results = model(image)
key_points_list, detections_list = results

# Convert to supervision format for filtering and visualization
key_points = key_points_list[0].to_supervision()

# Filter for "person" class (class_id=0) for COCO pose models
person_mask = key_points.class_id == 0
key_points_filtered = key_points[person_mask]

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

**Skeleton connections**: Access via `model.skeletons[class_id]`. The `model.skeletons` list has one element per detection class. For COCO pose models (single "person" class), use `model.skeletons[0]`. For custom multi-class models, each class has its own skeleton at the corresponding index.

Pass to `EdgeAnnotator`: `sv.EdgeAnnotator(edges=model.skeletons[0])` or omit for auto-detection.

