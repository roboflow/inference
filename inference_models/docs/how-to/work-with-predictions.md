# Work with Model Predictions

This guide shows you how to work with predictions from `inference-models`, including converting to different formats, accessing prediction data, filtering results, and visualizing outputs.

## Understanding Prediction Formats

### `inference-models` vs Supervision

`inference-models` uses its own prediction dataclasses that are **similar but not identical** to Supervision's format. This may seem confusing at first, but there are important reasons for this design:

**Why separate formats?**

- **Performance**: `inference-models` uses PyTorch tensors internally for maximum throughput
- **Minimal metadata**: Predictions carry only essential data, reducing memory overhead
- **Backend optimization**: Different backends (TensorRT, PyTorch, ONNX) can optimize tensor operations

**Easy conversion**: Use `.to_supervision()` to convert to Supervision format when you need rich visualization and analysis tools.

```python
from inference_models import AutoModel
import supervision as sv

model = AutoModel.from_pretrained("yolov8n-640")
predictions = model(image)

# inference-models format (lightweight, tensor-based)
result = predictions[0]  # ObjectDetectionResult

# Convert to Supervision format (rich features, numpy-based)
detections = result.to_supervision()  # sv.Detections
```

!!! tip "When to convert"
    - Keep `inference-models` format for high-throughput pipelines and filtering
    - Convert to Supervision when you need visualization, tracking, or advanced analysis

!!! warning "Work in Progress"
    The prediction format is evolving. We're working on making essential properties like `class_names` and other metadata transferrable to Supervision objects. However, these will likely use **references** to the model instance rather than duplicating data in each prediction.

    For example, `model.class_names` is the source of truth, rather than each prediction holding a separate copy of the class names list. This reduces memory overhead and ensures consistency.

### Prediction Types

Different model types return different prediction formats:

#### Supervision-Compatible Formats

These formats can be converted to Supervision using `.to_supervision()`:

| Model Type | `inference-models` Class | Supervision Class | Key Attributes |
|------------|--------------------------|-------------------|----------------|
| Object Detection | `ObjectDetectionResult` | `sv.Detections` | `xyxy`, `confidence`, `class_id`, `class_name` |
| Instance Segmentation | `InstanceSegmentationResult` | `sv.Detections` | Same as detection + `mask` |
| Keypoint Detection (Pose) | `KeypointsResult` | `sv.KeyPoints` | `xy`, `confidence`, `class_id` (person boxes + keypoints) |
| Keypoint Detection (Face) | `FaceDetectionResult` | `sv.Detections` | `xyxy`, `confidence`, `landmarks` (face boxes + landmarks) |

!!! note "Two Keypoint Detection Variants"
    - **Pose estimation** (YOLOv8-pose, YOLOv11-pose): Returns person bounding boxes with keypoints (e.g., nose, eyes, shoulders). Converts to `sv.KeyPoints`.
    - **Face detection** (MediaPipe Face): Returns face bounding boxes with facial landmarks. Converts to `sv.Detections` with landmarks in metadata.

#### Non-Supervision Formats

These formats have specialized structures and don't convert to Supervision:

| Model Type | Response Class | Key Attributes | Use Case |
|------------|----------------|----------------|----------|
| Classification | `ClassificationResult` | `top_class`, `confidence`, `predictions` (dict) | Image classification |
| Embeddings (CLIP) | `ClipEmbeddingResponse` | `embeddings` (List[List[float]]) | Similarity search, zero-shot |
| Embeddings (SAM) | `SamEmbeddingResponse` | `image_id`, cached embeddings | Interactive segmentation |
| Semantic Segmentation | `SemanticSegmentationResult` | `mask` (HxW class IDs), `class_names` | Pixel-level classification |
| Depth Estimation | `DepthEstimationResponse` | `normalized_depth` (HxW floats 0-1) | Depth maps |
| OCR | `OCRInferenceResponse` | `result` (text), `predictions` (word boxes) | Text extraction |
| Gaze Detection | `GazeDetectionInferenceResponse` | `face` (box + landmarks), `yaw`, `pitch` | Gaze direction |
| VLM/LMM | `LMMInferenceResponse` | `response` (str or dict) | Vision-language tasks |

## Working with `inference-models` Format

### Object Detection

```python
from inference_models import AutoModel
import cv2

# Load model and run inference
model = AutoModel.from_pretrained("yolov8n-640")
image = cv2.imread("path/to/image.jpg")
predictions = model(image)

# Access first image predictions (batch of 1)
result = predictions[0]  # ObjectDetectionResult

# Access tensor data (PyTorch tensors for performance)
print(f"Boxes (xyxy): {result.xyxy}")  # torch.Tensor shape [N, 4]
print(f"Confidence: {result.confidence}")  # torch.Tensor shape [N]
print(f"Class IDs: {result.class_id}")  # torch.Tensor shape [N]
print(f"Class names: {result.class_name}")  # List[str]

# Access model metadata (source of truth)
print(f"All classes: {model.class_names}")  # Reference to model's class list

# Iterate over detections
for i in range(len(result)):
    print(f"Detection {i}:")
    print(f"  Box: {result.xyxy[i]}")
    print(f"  Confidence: {result.confidence[i]:.2f}")
    print(f"  Class: {result.class_name[i]}")
    print(f"  Class ID: {result.class_id[i]} -> {model.class_names[result.class_id[i]]}")
```

### Classification

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("resnet50")
predictions = model(image)

# Access top prediction
result = predictions[0]
print(f"Top class: {result.top_class}")
print(f"Confidence: {result.confidence:.2f}")

# Access all class probabilities
for class_name, prob in result.predictions.items():
    print(f"{class_name}: {prob['confidence']:.2f}")
```

## Converting to Supervision Format

When you need rich visualization and analysis features, convert to [Supervision](https://supervision.roboflow.com) format using `.to_supervision()`:

### Object Detection

```python
import supervision as sv
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-640")
predictions = model(image)

# Convert to Supervision Detections
detections = predictions[0].to_supervision()

# Now you have sv.Detections with numpy arrays
print(f"Number of detections: {len(detections)}")
print(f"Boxes: {detections.xyxy}")  # numpy array [[x1, y1, x2, y2], ...]
print(f"Confidence: {detections.confidence}")  # numpy array [0.95, 0.87, ...]
print(f"Class IDs: {detections.class_id}")  # numpy array [0, 1, 2, ...]
print(f"Class names: {detections.data['class_name']}")  # numpy array ['person', 'car', ...]
```

!!! note "Format Differences"
    - `inference-models`: Uses PyTorch tensors (`torch.Tensor`) for speed, references model metadata
    - Supervision: Uses NumPy arrays (`np.ndarray`) for compatibility, includes metadata in object

    The `.to_supervision()` method handles the conversion automatically, copying necessary metadata from the model instance.

### Instance Segmentation

```python
model = AutoModel.from_pretrained("yolov8n-seg-640")
predictions = model(image)

# Convert to Supervision Detections (includes masks)
detections = predictions[0].to_supervision()

# Access masks
print(f"Masks shape: {detections.mask.shape}")  # (N, H, W)
print(f"Has masks: {detections.mask is not None}")
```

## Filtering Predictions

### By Confidence Threshold

```python
import supervision as sv

detections = predictions[0].to_supervision()

# Filter by confidence
high_confidence = detections[detections.confidence > 0.7]
print(f"High confidence detections: {len(high_confidence)}")
```

### By Class

```python
# Filter by class ID
person_detections = detections[detections.class_id == 0]

# Filter by class name (if class_name is in data)
if 'class_name' in detections.data:
    person_detections = detections[detections.data['class_name'] == 'person']
```

### By Area

```python
import numpy as np

# Calculate areas
areas = detections.box_area

# Filter by area
large_objects = detections[areas > 10000]  # pixels
```

## Visualizing Predictions

### Basic Annotation

```python
import cv2
import supervision as sv
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-640")
image = cv2.imread("path/to/image.jpg")
predictions = model(image)
detections = predictions[0].to_supervision()

# Annotate with bounding boxes
box_annotator = sv.BoxAnnotator()
annotated = box_annotator.annotate(image.copy(), detections)

cv2.imwrite("output.jpg", annotated)


### Save as CSV

```python
import pandas as pd

df = pd.DataFrame({
    "x1": detections.xyxy[:, 0],
    "y1": detections.xyxy[:, 1],
    "x2": detections.xyxy[:, 2],
    "y2": detections.xyxy[:, 3],
    "confidence": detections.confidence,
    "class_id": detections.class_id,
})

df.to_csv("predictions.csv", index=False)
```

## Batch Processing

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("yolov8n-640")

# Load multiple images
images = [cv2.imread(f"image_{i}.jpg") for i in range(5)]

# Batch inference
predictions = model(images)

# Process each result
for i, pred in enumerate(predictions):
    detections = pred.to_supervision()
    print(f"Image {i}: {len(detections)} detections")
```

## Advanced Use Cases

### Tracking Objects

```python
import supervision as sv

tracker = sv.ByteTrack()

for frame in video_frames:
    predictions = model(frame)
    detections = predictions[0].to_supervision()
    detections = tracker.update_with_detections(detections)
```

### Crop Detections

```python
detections = predictions[0].to_supervision()

for i in range(len(detections)):
    x1, y1, x2, y2 = detections.xyxy[i].astype(int)
    crop = image[y1:y2, x1:x2]
    cv2.imwrite(f"crop_{i}.jpg", crop)
```

## Next Steps

- [Supervision Documentation](https://supervision.roboflow.com) - Learn more about Supervision
- [Choose the Right Backend](choose-backend.md) - Optimize performance
- [Supported Models](../models/index.md) - Browse available models

```python
import supervision as sv

# Create annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Create labels
labels = [
    f"{class_name} {confidence:.2f}"
    for class_name, confidence in zip(
        detections.data['class_name'],
        detections.confidence
    )
]

# Annotate
annotated = box_annotator.annotate(image.copy(), detections)
annotated = label_annotator.annotate(annotated, detections, labels=labels)

cv2.imwrite("output_with_labels.jpg", annotated)
```

### Instance Segmentation Visualization

```python
# Annotate with masks
mask_annotator = sv.MaskAnnotator()
label_annotator = sv.LabelAnnotator()

annotated = mask_annotator.annotate(image.copy(), detections)
annotated = label_annotator.annotate(annotated, detections)

cv2.imwrite("segmentation_output.jpg", annotated)
```

## Saving Predictions

### Save as JSON

```python
import json

# Convert detections to dict
results = {
    "boxes": detections.xyxy.tolist(),
    "confidence": detections.confidence.tolist(),
    "class_id": detections.class_id.tolist(),
}

with open("predictions.json", "w") as f:
    json.dump(results, f, indent=2)
```

