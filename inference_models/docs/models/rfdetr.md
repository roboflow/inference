# RFDetr

RFDetr (Roboflow Detection Transformer) is a state-of-the-art object detection model based on the DETR architecture with significant improvements for production deployment.

## Overview

- **Task**: Object Detection
- **Architecture**: Detection Transformer (DETR)
- **Backends**: PyTorch, TensorRT
- **License**: [Apache 2.0](https://github.com/roboflow/inference/blob/main/inference_models/models/rfdetr/LICENSE.txt)
- **Access**: Public (no API key required)

## Model Variants

### Available on Roboflow Platform

| Model ID | Parameters | Input Size | COCO mAP | Speed (ms) | Use Case |
|----------|------------|------------|----------|------------|----------|
| `rfdetr-nano` | ~10M | 640×640 | ~40% | ~15ms | Edge devices, real-time |
| `rfdetr-small` | ~25M | 640×640 | ~45% | ~20ms | Balanced performance |
| `rfdetr-base` | ~50M | 640×640 | ~50% | ~30ms | High accuracy |
| `rfdetr-medium` | ~75M | 640×640 | ~52% | ~40ms | Very high accuracy |
| `rfdetr-large` | ~100M | 640×640 | ~54% | ~50ms | Maximum accuracy |

All variants are trained on the COCO dataset with 80 object classes.

## Installation

### Basic Installation

```bash
# CPU (PyTorch only)
pip install inference-models

# GPU with TensorRT support
pip install "inference-models[torch-cu128,trt10]" "tensorrt==10.12.0.36"
```

## Quick Start

### Basic Usage

```python
from inference_models import AutoModel
import cv2

# Load model
model = AutoModel.from_pretrained("rfdetr-base")

# Load image
image = cv2.imread("path/to/image.jpg")

# Run inference
predictions = model(image)

# Access results
for detection in predictions[0]:
    print(f"Class: {detection.class_id}, Confidence: {detection.confidence:.2f}")
    print(f"Box: {detection.xyxy}")
```

### With Supervision

```python
from inference_models import AutoModel
import cv2
import supervision as sv

# Load model
model = AutoModel.from_pretrained("rfdetr-base")

# Load image
image = cv2.imread("path/to/image.jpg")

# Run inference
predictions = model(image)

# Convert to Supervision format
detections = predictions[0].to_supervision()

# Annotate
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotated = box_annotator.annotate(image.copy(), detections)
annotated = label_annotator.annotate(annotated, detections)

# Save
cv2.imwrite("output.jpg", annotated)
```

## Model Interface

### Input

- **Type**: `np.ndarray` or `List[np.ndarray]`
- **Format**: BGR (OpenCV default) or RGB
- **Shape**: Any size (automatically resized to 640×640)
- **Batch**: Supports single image or batch of images

### Output

Returns `List[Detections]` where each `Detections` object contains:

```python
@dataclass
class Detections:
    xyxy: torch.Tensor          # (n_boxes, 4) - bounding boxes
    class_id: torch.Tensor      # (n_boxes,) - class indices
    confidence: torch.Tensor    # (n_boxes,) - confidence scores
    image_metadata: Optional[dict] = None
    bboxes_metadata: Optional[List[dict]] = None
```

### Methods

```python
# Run inference
predictions = model(images)

# Get class names
class_names = model.class_names

# Convert to Supervision
sv_detections = predictions[0].to_supervision()
```

## Advanced Usage

### Batch Processing

```python
import cv2

# Load multiple images
images = [
    cv2.imread("image1.jpg"),
    cv2.imread("image2.jpg"),
    cv2.imread("image3.jpg"),
]

# Batch inference
predictions = model(images)

# Process each result
for i, pred in enumerate(predictions):
    print(f"Image {i}: {len(pred.xyxy)} detections")
```

### Confidence Filtering

```python
# Run inference
predictions = model(image)

# Filter by confidence
high_conf = predictions[0].confidence > 0.5
filtered_boxes = predictions[0].xyxy[high_conf]
filtered_classes = predictions[0].class_id[high_conf]
filtered_conf = predictions[0].confidence[high_conf]
```

### Specify Backend

```python
# Force PyTorch backend
model = AutoModel.from_pretrained(
    "rfdetr-base",
    backend_type="torch"
)

# Force TensorRT backend (if available)
model = AutoModel.from_pretrained(
    "rfdetr-base",
    backend_type="trt"
)
```

### Custom Device

```python
# Use specific GPU
model = AutoModel.from_pretrained(
    "rfdetr-base",
    device="cuda:1"
)

# Use CPU
model = AutoModel.from_pretrained(
    "rfdetr-base",
    device="cpu"
)
```

## COCO Classes

RFDetr models are trained on COCO with 80 classes:

```python
model = AutoModel.from_pretrained("rfdetr-base")
print(model.class_names)
# ['person', 'bicycle', 'car', 'motorcycle', 'airplane', ...]
```

Full class list: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

## Performance Tips

1. **Use TensorRT for production**: 2-3x faster than PyTorch
2. **Batch processing**: Process multiple images together for better throughput
3. **Choose appropriate variant**: Nano for edge, Base for balanced, Large for accuracy
4. **Pre-allocate memory**: Reuse model instance for multiple inferences

## Troubleshooting

### Model fails to load

Ensure you have the required backend installed:

```bash
# For PyTorch
pip install inference-models

# For TensorRT
pip install "inference-models[trt10]" "tensorrt==10.12.0.36"
```

### Slow inference

- Use TensorRT backend for GPU
- Enable batch processing
- Use smaller variant (nano/small)

## Citation

```bibtex
@article{rfdetr2024,
  title={RFDetr: Roboflow Detection Transformer},
  author={Roboflow Team},
  year={2024}
}
```

## Related Models

- [YOLOv8](yolov8.md) - Faster inference, similar accuracy
- [YOLOv11](yolov11.md) - Latest YOLO version
- [YOLO-NAS](yolonas.md) - Neural Architecture Search based detector

## Next Steps

- [Quick Overview](../getting-started/overview.md)
- [Object Detection Tutorial](../how-to/object-detection.md)
- [API Reference](../api-reference/)

