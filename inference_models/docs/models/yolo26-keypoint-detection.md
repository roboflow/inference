# YOLO26 - Keypoint Detection

YOLO26 is the latest addition to the Ultralytics YOLO model series. The keypoint detection variant (also known as pose estimation) detects objects and predicts keypoints for each detected instance, commonly used for human pose estimation.

## Overview

YOLO26 for keypoint detection combines object detection with keypoint localization. Key features include:

- **NMS-free end-to-end inference** - Removing non-maximum suppression helps achieve lower inference latencies.
- **DFL removal** - Distribution Focal Loss removed for simpler export and broader edge compatibility
- **Residual Log-Likelihood Estimation (RLE)** - More accurate keypoint localization
- **Multi-person support** - Detects and tracks keypoints for multiple instances simultaneously
- **Multiple model sizes** - From nano to extra-large variants

## License

**AGPL-3.0**

!!! info "Commercial Licensing"
    - **AGPL-3.0**: Free for open-source projects. Requires derivative works to be open-sourced.
    - **Paid Roboflow customers**: Automatically get access to use any YOLO26 models trained on or uploaded to the Roboflow platform for commercial use.
    - **Free Roboflow customers**: Can use YOLO26 via the serverless hosted API, or commercially self-hosted with a paid plan.

    Learn more: [Roboflow Licensing](https://roboflow.com/licensing) | [YOLO26 License Details](https://roboflow.com/model-licenses/yolo26)

## Pre-trained Model IDs

All pre-trained YOLO26 keypoint detection models are trained on the COCO dataset (17 keypoints for human pose) and are **open access** (no API key required).

| Model Size | 640×640 |
|------------|---------|
| Nano | `yolo26n-pose` |
| Small | `yolo26s-pose` |
| Medium | `yolo26m-pose` |
| Large | `yolo26l-pose` |
| Extra-Large | `yolo26x-pose` |

**COCO Keypoints**: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

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

# Filter for specific class if needed (class_id=0 for single-class models)
target_mask = key_points.class_id == 0
key_points_filtered = key_points[target_mask]

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

**Skeleton connections**: Access via `model.skeletons[class_id]`. The `model.skeletons` list has one element per detection class. For single-class models, use `model.skeletons[0]`. For custom multi-class models, each class has its own skeleton at the corresponding index.

Pass to `EdgeAnnotator`: `sv.EdgeAnnotator(edges=model.skeletons[0])` or omit for auto-detection.
