# OWLv2 - Open-Vocabulary Object Detection

OWLv2 (Open-World Localization v2) is an open-vocabulary object detection model developed by Google Research that can detect objects using text prompts or visual examples.

## Overview

OWLv2 is a vision transformer-based model that enables zero-shot and few-shot object detection. Key capabilities include:

- **Text-Prompted Detection** - Detect objects using natural language descriptions
- **Visual Example Detection** - Detect objects by providing visual examples (few-shot learning)
- **Zero-Shot Detection** - No training required for new object classes
- **Open Vocabulary** - Works with arbitrary object classes

!!! warning "Visual Examples Recommended"
    The current implementation in `inference-models` is optimized for visual example-based detection (few-shot learning). Text-only prompting may have limited support.

!!! info "License & Attribution"
    **License**: Apache 2.0<br>**Source**: [Google Research](https://github.com/google-research/scenic/tree/main/scenic/projects/owl_vit)<br>**Paper**: [Scaling Open-Vocabulary Object Detection](https://arxiv.org/abs/2306.09683)

## Pre-trained Model IDs

OWLv2 pre-trained models are available and **require a Roboflow API key**.

| Model ID | Description |
|----------|-------------|
| `google/owlv2-base-patch14-ensemble` | Base ensemble model - balanced performance |
| `google/owlv2-large-patch14-ensemble` | Large ensemble model - higher accuracy, slower |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `hf` (Hugging Face) | Included in base installation |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not available for training |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via OWLv2 block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Usage Examples

### Text-Based Detection (Open Vocabulary)

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load model (requires API key)
model = AutoModel.from_pretrained("google/owlv2-base-patch14-ensemble", api_key="your_roboflow_api_key")

# Load image
image = cv2.imread("path/to/image.jpg")

# Detect objects with text prompts
predictions = model(image, classes=["dog", "person", "car"])

detections = predictions[0].to_supervision()

# Annotate image
bounding_box_annotator = sv.BoxAnnotator()
annotated_image = bounding_box_annotator.annotate(image, detections)

cv2.imwrite("annotated.jpg", annotated_image)
```

### Few-Shot Detection with Visual Examples

```python
import cv2
from inference_models import AutoModel
from inference_models.models.owlv2.entities import ReferenceExample, ReferenceBoundingBox

# Load model
model = AutoModel.from_pretrained("google/owlv2-base-patch14-ensemble", api_key="your_roboflow_api_key")

# Load images
image = cv2.imread("path/to/image.jpg")
reference_image = cv2.imread("path/to/reference.jpg")

# Define reference examples with bounding boxes
reference_examples = [
    ReferenceExample(
        image=reference_image,
        boxes=[
            ReferenceBoundingBox(x=100, y=50, w=80, h=80, cls="logo"),
            ReferenceBoundingBox(x=300, y=200, w=120, h=150, cls="product"),
        ],
    )
]

# Detect similar objects
predictions = model.infer_with_reference_examples(
    image,
    reference_examples=reference_examples,
    confidence_threshold=0.99,
    iou_threshold=0.3
)

detections = predictions[0].to_supervision()
print(f"Found {len(detections)} objects")
```

### Using Embeddings Cache for Better Performance

```python
import cv2
from inference_models import AutoModel
from inference_models.models.owlv2.cache import (
    OwlV2ClassEmbeddingsCache,
    OwlV2ImageEmbeddingsCache,
)

# Create cache instances
class_embeddings_cache = OwlV2ClassEmbeddingsCache()
image_embeddings_cache = OwlV2ImageEmbeddingsCache()

# Load model with caching enabled
model = AutoModel.from_pretrained(
    "google/owlv2-base-patch14-ensemble",
    api_key="your_roboflow_api_key",
    owlv2_class_embeddings_cache=class_embeddings_cache,
    owlv2_images_embeddings_cache=image_embeddings_cache,
)

# First inference - embeddings will be cached
image = cv2.imread("path/to/image.jpg")
predictions = model(image, classes=["dog", "person"])

# Subsequent inferences with same image/classes will be faster
predictions = model(image, classes=["dog", "person"])  # Uses cached embeddings
```

## Workflows Integration

OWLv2 can be used in Roboflow Workflows for complex computer vision pipelines. The OWLv2 block supports both text prompts and visual examples for flexible object detection.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - OWLv2 requires GPU for acceptable performance
2. **Start with high confidence for few-shot** - When using reference examples, start with `confidence_threshold=0.99` and adjust down if needed
3. **Provide good examples** - For few-shot learning, provide clear, representative bounding box examples
4. **Choose the right model** - Use `base-patch14-ensemble` for speed, `large-patch14-ensemble` for accuracy
5. **Cache embeddings** - The model automatically caches embeddings for faster repeated inference

## Few-Shot Learning Best Practices

When using visual examples:

- ✅ **Provide clear examples**: Use well-lit, unoccluded objects
- ✅ **Multiple examples help**: Provide 2-3 examples per class when possible
- ✅ **Consistent examples**: Use examples similar to target objects
- ❌ **Avoid poor quality**: Don't use blurry or partially visible examples
- ❌ **Avoid extreme variations**: Keep examples consistent in appearance

## Common Use Cases

- **Product Detection** - Detect products using example images
- **Logo Detection** - Find logos by providing reference examples
- **Custom Object Detection** - Detect specialized objects without training
- **Prototype Development** - Quickly test detection ideas before full training

