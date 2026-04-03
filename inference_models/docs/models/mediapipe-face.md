# MediaPipe Face Detection

MediaPipe Face Detection is a lightweight face detection model developed by Google. It detects faces and predicts 6 facial keypoints (eyes, nose, mouth, ears) optimized for mobile and edge devices.

## Overview

**Resources**: [MediaPipe Documentation](https://developers.google.com/mediapipe/solutions/vision/face_detector) | [GitHub Repository](https://github.com/google/mediapipe)

MediaPipe Face Detection provides fast and efficient face detection with keypoint localization. Key features include:

- **Lightweight architecture** - Optimized for mobile and edge deployment
- **6 facial keypoints** - Eyes, nose, mouth, and ears
- **Fast inference** - Efficient processing on CPU
- **Single-stage detection** - Detects faces and keypoints simultaneously
- **TensorFlow Lite backend** - Efficient mobile-optimized runtime

## License

**Apache 2.0**

!!! info "Open Source License"
    MediaPipe Face Detection is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

MediaPipe Face Detection has one pre-trained model available via the Roboflow API and **requires a Roboflow API key**.

!!! info "Getting a Roboflow API Key"
    To use MediaPipe models, you'll need a [Roboflow account](https://app.roboflow.com/) (free) and [API key](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key).

| Model | Model ID | Use Case |
|-------|----------|----------|
| Face Detector | `mediapipe/face-detector` | Face detection with 6 keypoints |

**Detected keypoints**: right-eye, left-eye, nose, mouth, right-ear, left-ear

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `mediapipe` | `mediapipe` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Pre-trained model only |
| **Upload Weights** | ❌ Pre-trained model only |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Keypoint Detection block |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model (requires Roboflow API key)
model = AutoModel.from_pretrained(
    "mediapipe/face-detector",
    api_key="your_roboflow_api_key"
)
image = cv2.imread("path/to/image.jpg")

# Run inference - returns tuple of (List[KeyPoints], List[Detections])
results = model(image)
key_points_list, detections_list = results

# Convert to supervision format for visualization
key_points = key_points_list[0].to_supervision()
detections = detections_list[0].to_supervision()

# Annotate image with bounding boxes and keypoints
box_annotator = sv.BoxAnnotator()
vertex_annotator = sv.VertexAnnotator()

# EdgeAnnotator can use skeleton from model or auto-detect based on keypoint count
edge_annotator = sv.EdgeAnnotator(edges=model.skeletons[0])

annotated_image = box_annotator.annotate(image.copy(), detections)
annotated_image = edge_annotator.annotate(annotated_image, key_points)
annotated_image = vertex_annotator.annotate(annotated_image, key_points)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```

## Output Format

Returns: `Tuple[List[KeyPoints], List[Detections]]` (one element per image in batch)

**Keypoint names**: right-eye, left-eye, nose, mouth, right-ear, left-ear (6 keypoints)

**Skeleton connections**: Access via `model.skeletons[0]` which returns `[(5, 1), (1, 2), (4, 0), (0, 2), (2, 3)]`. The `model.skeletons` list has one element per detection class; since this model has only one class (face), always use index 0.

Pass to `EdgeAnnotator`: `sv.EdgeAnnotator(edges=model.skeletons[0])` or omit for auto-detection.

## Use Cases

MediaPipe Face Detection is ideal for:

- ✅ **Face detection** - Locate faces in images
- ✅ **Facial landmark detection** - Get key facial feature positions
- ✅ **Mobile/edge deployment** - Lightweight model for resource-constrained devices
- ✅ **Fast processing** - Efficient CPU inference
- ✅ **Preprocessing for face analysis** - Detect faces before running face recognition or analysis

