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
result = predictions[0]  # Detections

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
| Object Detection | `Detections` | `sv.Detections` | `xyxy`, `confidence`, `class_id` |
| Instance Segmentation | `InstanceDetections` | `sv.Detections` | Same as detection + `mask` |
| Keypoint Detection | `KeyPoints` | `sv.KeyPoints` | `xy`, `confidence`, `class_id` |

#### Other Formats

These formats have specialized structures:

| Model Type | `inference-models` Class | Key Attributes |
|------------|--------------------------|----------------|
| Classification | `ClassificationPrediction` | `class_id`, `confidence` |
| Multi-Label Classification | `MultiLabelClassificationPrediction` | `class_ids`, `confidence` |
| Semantic Segmentation | `SemanticSegmentationResult` | `segmentation_map`, `confidence` |

Other supported formats include embeddings (CLIP, SAM), depth estimation, OCR, gaze detection, and vision-language models (VLM/LMM).

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
result = predictions[0]  # Detections

# Access tensor data (PyTorch tensors for performance)
print(f"Boxes (xyxy): {result.xyxy}")  # torch.Tensor shape [N, 4]
print(f"Confidence: {result.confidence}")  # torch.Tensor shape [N]
print(f"Class IDs: {result.class_id}")  # torch.Tensor shape [N]

# Access model metadata (source of truth)
print(f"All classes: {model.class_names}")  # Reference to model's class list

# Iterate over detections
for i in range(len(result.xyxy)):
    print(f"Detection {i}:")
    print(f"  Box: {result.xyxy[i]}")
    print(f"  Confidence: {result.confidence[i]:.2f}")
    print(f"  Class ID: {result.class_id[i]}")
    print(f"  Class name: {model.class_names[result.class_id[i]]}")
```

### Classification

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained("resnet50")
predictions = model(image)

# Access prediction
result = predictions[0]  # ClassificationPrediction
print(f"Class ID: {result.class_id}")  # torch.Tensor
print(f"Confidence: {result.confidence:.2f}")  # torch.Tensor
print(f"Class name: {model.class_names[result.class_id]}")
```

### Multi-Label Classification

```python
model = AutoModel.from_pretrained("multi-label-model")
predictions = model(image)

result = predictions[0]  # MultiLabelClassificationPrediction
print(f"Predicted class IDs: {result.class_ids}")  # torch.Tensor
print(f"Confidences: {result.confidence}")  # torch.Tensor
for class_id, conf in zip(result.class_ids, result.confidence):
    print(f"{model.class_names[class_id]}: {conf:.2f}")
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
from inference_models import AutoModel

# Run inference
model = AutoModel.from_pretrained("yolov8n-640")
predictions = model(image)

# Convert to Supervision format
detections = predictions[0].to_supervision()  # sv.Detections

# Filter by confidence threshold
high_confidence = detections[detections.confidence > 0.7]
print(f"High confidence detections: {len(high_confidence)}")
```

### By Bounding Box Area

```python
import supervision as sv
from inference_models import AutoModel

# Run inference
model = AutoModel.from_pretrained("yolov8n-640")
predictions = model(image)

# Convert to Supervision format
detections = predictions[0].to_supervision()  # sv.Detections

# Calculate areas and filter
areas = detections.box_area
large_objects = detections[areas > 10000]  # Filter objects larger than 10000 pixels
print(f"Large objects: {len(large_objects)}")
```

## Visualization

Use Supervision's annotators to visualize predictions:

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model and run inference
model = AutoModel.from_pretrained("yolov8n-640")
image = cv2.imread("path/to/image.jpg")
predictions = model(image)

# Convert to Supervision format
detections = predictions[0].to_supervision()  # sv.Detections

# Create annotator and visualize
box_annotator = sv.BoundingBoxAnnotator()
annotated_image = box_annotator.annotate(
    scene=image.copy(),
    detections=detections
)

# Save result
cv2.imwrite("output.jpg", annotated_image)
```

## Next Steps

- [Supervision Documentation](https://supervision.roboflow.com) - Learn more about Supervision annotators and utilities
- [Choose the Right Backend](choose-backend.md) - Optimize performance for your use case
- [Supported Models](../models/index.md) - Browse available models

