# RF-DETR - Keypoint Detection

RF-DETR is a real-time keypoint detection model developed by Roboflow. 

## Overview

RF-DETR features:

- **State-of-the-art accuracy** - Leading performance on COCO and real-world benchmarks
- **Exceptional domain transfer** - Designed to excel across diverse domains and dataset sizes
- **Multi-class support** - Can be trained to learn keypoints of objects of different classes, with different number of keypoints per class
- **Localization uncertainty** - Predicts a per-keypoint covariance (positional uncertainty), visualizable as confidence ellipses
- **Real-time performance** - Optimized for speed without sacrificing accuracy
- **Production-ready** - Built for deployment on edge devices and cloud infrastructure

## License

**Apache 2.0**

RF-DETR is released under the Apache 2.0 license, making it free for both commercial and non-commercial use.

Learn more: [Apache 2.0 License](https://github.com/roboflow/inference/blob/main/inference_models/models/rfdetr/LICENSE.txt)

## Pre-trained Model IDs

RF-DETR Keypoint Detection preview model is trained on the COCO keypoints dataset (17 keypoints for human pose) and is **open access** (no API key required).

| Model Size | Model ID |
|------------|----------|
| Preview | `rfdetr-keypoint-preview` |

**COCO Keypoints**: nose, left_eye, right_eye, left_ear, right_ear, left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip, right_hip, left_knee, right_knee, left_ankle, right_ankle

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Installation

Install with one of the following extras depending on your backend:

- **ONNX**: `onnx-cpu`, `onnx-cu12`

## Usage Example

### Using Pre-trained Models

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

# Optionally overlay per-keypoint uncertainty ellipses (see "Keypoint Uncertainty" below)
ellipse_annotator = sv.VertexEllipseAreaAnnotator()
annotated_image = ellipse_annotator.annotate(annotated_image, key_points_filtered)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```

## Output Format

Returns: `Tuple[List[KeyPoints], Optional[List[Detections]]]` (one element per image in batch)

**Skeleton connections**: Access via `model.skeletons[class_id]`. The `model.skeletons` list has one element per detection class. For single-class models, use `model.skeletons[0]`. For custom multi-class models, each class has its own skeleton at the corresponding index.

Pass to `EdgeAnnotator`: `sv.EdgeAnnotator(edges=model.skeletons[0])` or omit for auto-detection.

## Keypoint Uncertainty (Covariance)

Unlike most keypoint detectors, RF-DETR predicts how *certain* it is about each keypoint's
location. The model emits a 2D Gaussian per keypoint, which `inference-models` converts into a
pixel-space covariance matrix (`2x2`) describing the positional uncertainty in the original image.

The covariance is available on the returned `KeyPoints` object and is carried into the Supervision
format under `data["covariance"]`:

```python
key_points_list, _ = model(image)
key_points = key_points_list[0]

# Per-keypoint covariance, shape (num_instances, num_keypoints, 2, 2)
print(key_points.covariance.shape)

# After conversion, it lives under data["covariance"] (same shape)
sv_key_points = key_points.to_supervision()
print(sv_key_points.data["covariance"].shape)
```

Notes:

- Shape is `(num_instances, num_keypoints, 2, 2)`, aligned with `key_points.xy`.
- Each matrix is symmetric, in **original-image pixel units**, and the diagonal holds the
  x/y variances. Larger ellipses mean the model is less certain about that keypoint.
- Keypoints that are filtered out (below the keypoint threshold, or unused slots for classes with
  fewer keypoints) have their covariance set to `NaN`.
- Only RF-DETR populates covariance; other keypoint models leave it as `None`, and
  `to_supervision()` omits the `data["covariance"]` key in that case.

### Visualizing uncertainty ellipses

Supervision provides covariance-aware annotators that read `data["covariance"]`:

```python
import supervision as sv

# Filled, semi-transparent ellipses at 1σ/2σ/3σ levels
ellipse_annotator = sv.VertexEllipseAreaAnnotator()
annotated = ellipse_annotator.annotate(image.copy(), sv_key_points)

# Stroke-only rings, or a faded halo, are also available:
# sv.VertexEllipseOutlineAnnotator()
# sv.VertexEllipseHaloAnnotator()
```
