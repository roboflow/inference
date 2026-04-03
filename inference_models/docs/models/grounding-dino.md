# Grounding DINO - Zero-Shot Object Detection

Grounding DINO is a zero-shot object detection model developed by IDEA Research that can detect objects in images using arbitrary text prompts.

## Overview

Grounding DINO combines the power of DINO (a self-supervised vision transformer) with grounded pre-training to enable open-vocabulary object detection. Key capabilities include:

- **Text-Prompted Detection** - Detect objects using natural language descriptions
- **Zero-Shot Detection** - No training required for new object classes
- **High Accuracy** - State-of-the-art performance on open-vocabulary detection benchmarks
- **Flexible Prompting** - Accepts single words, phrases, or detailed descriptions

!!! note "Best for Common Objects"
    Grounding DINO is most effective at identifying common objects (e.g., cars, people, dogs). It is less effective at identifying uncommon or highly specific objects (e.g., a specific type of car, a specific person).

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [IDEA Research](https://github.com/IDEA-Research/GroundingDINO)<br>**Paper**: [Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection](https://arxiv.org/abs/2303.05499)

## Pre-trained Model IDs

Grounding DINO pre-trained models are available and **require a Roboflow API key**.

| Model ID | Description |
|----------|-------------|
| `grounding-dino` | Base model - balanced performance |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not available for training |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via Grounding DINO block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Usage Examples

### Basic Object Detection

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model (requires API key)
model = AutoModel.from_pretrained("grounding-dino", api_key="your_roboflow_api_key")

# Load image
image = cv2.imread("path/to/image.jpg")

# Detect objects with text prompts
predictions = model(image, ["person", "car", "dog"], conf_thresh=0.35)
detections = predictions[0].to_supervision()

# Annotate image
bounding_box_annotator = sv.BoxAnnotator()
annotated_image = bounding_box_annotator.annotate(image, detections)

# Save or display
cv2.imwrite("annotated.jpg", annotated_image)
```

### Using Phrase Descriptions

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("grounding-dino", api_key="your_roboflow_api_key")
image = cv2.imread("path/to/image.jpg")

# Use detailed phrase descriptions
predictions = model(
    image,
    ["red car", "person wearing hat", "small dog"],
    conf_thresh=0.3
)

print(f"Found {len(predictions[0].xyxy)} objects")
```

### Adjusting Detection Thresholds

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("grounding-dino", api_key="your_roboflow_api_key")
image = cv2.imread("path/to/image.jpg")

# Adjust confidence threshold for more/fewer detections
predictions = model(
    image,
    ["person", "vehicle"],
    conf_thresh=0.25,  # Lower threshold = more detections
    text_thresh=0.25   # Text matching threshold
)

detections = predictions[0].to_supervision()
print(f"Detected {len(detections)} objects")
```

## Workflows Integration

Grounding DINO can be used in Roboflow Workflows for complex computer vision pipelines. The Grounding DINO block accepts text prompts and returns object detections that can be combined with other blocks.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - Grounding DINO requires GPU for acceptable performance
2. **Optimize prompts** - Use clear, specific descriptions for better results
3. **Adjust thresholds** - Experiment with `conf_thresh` and `text_thresh` for your use case
4. **Batch processing** - Process multiple images together when possible
5. **Choose the right model** - Use `tiny` for speed, `base` for accuracy

## Prompting Best Practices

- ✅ **Use simple, common words**: "car", "person", "dog"
- ✅ **Be specific when needed**: "red car", "person wearing hat"
- ✅ **Use singular nouns**: "car" instead of "cars"
- ❌ **Avoid overly complex descriptions**: "a blue sedan parked on the street"
- ❌ **Avoid rare or technical terms**: Use "car" instead of "automobile"
