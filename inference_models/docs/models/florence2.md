# Florence-2 - Vision Language Model

Florence-2 is a versatile vision-language model developed by Microsoft Research that can perform a wide range of vision tasks through natural language prompts.

## Overview

Florence-2 is a unified, prompt-based model capable of handling diverse computer vision and vision-language tasks:

- **Object Detection** - Detect and locate objects in images
- **Instance Segmentation** - Segment individual object instances
- **Image Captioning** - Generate descriptive captions for images
- **Optical Character Recognition (OCR)** - Extract text from images
- **Phrase Grounding** - Locate objects based on text descriptions
- **Region Captioning** - Generate captions for specific image regions
- **Open Vocabulary Detection** - Detect objects from custom class lists

!!! warning "GPU Recommended"
    Florence-2 works best with GPU acceleration. CPU inference may be very slow or may not work properly for larger models.

!!! info "License & Attribution"
    **License**: MIT<br>**Source**: [Microsoft Research](https://github.com/microsoft/Florence)<br>**Paper**: [Florence-2: Advancing a Unified Representation for a Variety of Vision Tasks](https://arxiv.org/abs/2311.06242)

## Pre-trained Model IDs

Florence-2 pre-trained models are available and do **not** require a Roboflow API key.

| Model ID | Description |
|----------|-------------|
| `florence-2-base` | Base model (0.23B parameters) - faster inference |
| `florence-2-large` | Large model (0.77B parameters) - better accuracy |

You can also use fine-tuned models from Roboflow by specifying `project/version` as the model ID (requires API key).

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ LoRA fine-tuning only ([Guide](https://blog.roboflow.com/fine-tune-florence-2-object-detection/)) |
| **Upload Weights** | ✅ Upload fine-tuned models |
| **Serverless API (v2)** | ⚠️ Limited support (not yet fully stable) |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via Florence-2 block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Training & Fine-tuning

Florence-2 supports **LoRA (Low-Rank Adaptation) fine-tuning only** on the Roboflow platform. This allows you to adapt the model to your specific use case without training the entire model.

### When to Fine-tune Florence-2

Fine-tuning Florence-2 is beneficial when you need:

- **Domain-specific object detection** - Detect specialized objects in your industry (medical, industrial, etc.)
- **Custom captioning style** - Generate captions that match your specific terminology or format
- **Specialized OCR** - Improve text recognition for domain-specific fonts or layouts
- **Task-specific performance** - Optimize for a particular vision task with your data

### Recommended Use Cases for Fine-tuning

- ✅ **Medical imaging** - Detect anatomical structures or abnormalities
- ✅ **Industrial inspection** - Identify defects or components in manufacturing
- ✅ **Document analysis** - Extract structured data from forms or receipts
- ✅ **Retail** - Detect products or analyze shelf layouts
- ✅ **Agriculture** - Identify crops, pests, or plant diseases

Learn more: [How to Fine-tune Florence-2](https://blog.roboflow.com/fine-tune-florence-2-object-detection/)

## Supported Tasks

Florence-2 supports multiple vision tasks. Use the high-level API methods for each task:

| Task | Method | When to Use |
|------|--------|-------------|
| Image Captioning | `caption_image()` | Generate natural language descriptions of images at different levels of detail (normal, detailed, very detailed) |
| Object Detection | `detect_objects()` | Detect and locate all objects in an image, or detect specific classes using open vocabulary detection |
| OCR | `ocr_image()` | Extract all text from images without location information |
| OCR with Detection | `parse_document()` | Extract text with bounding box locations - ideal for document parsing and structured text extraction |
| Phrase Grounding | `ground_phrase()` | Locate objects in an image based on a text description (e.g., "red car", "person wearing hat") |
| Instance Segmentation | `segment_phrase()` | Generate segmentation masks for objects matching a text description |
| Region Segmentation | `segment_region()` | Generate segmentation masks for objects within specified bounding boxes |
| Region Classification | `classify_image_region()` | Classify what object is in a specific region of the image |
| Region Captioning | `caption_image_region()` | Generate captions for specific regions of the image |
| Region OCR | `ocr_image_region()` | Extract text from specific regions of the image |

## Usage Examples

### Image Captioning

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("florence-2-base")

# Load image
image = cv2.imread("path/to/image.jpg")

# Generate caption (normal, detailed, or very_detailed)
captions = model.caption_image(images=image, granularity="detailed")
print(f"Caption: {captions[0]}")
```

### Object Detection

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("florence-2-base")
image = cv2.imread("path/to/image.jpg")

# Detect all objects with class labels
detections = model.detect_objects(images=image, labels_mode="classes")
print(detections[0])

# Detect specific classes (open vocabulary)
detections = model.detect_objects(
    images=image,
    classes=["person", "car", "dog"]
)
print(detections[0])
```

### OCR (Text Extraction)

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("florence-2-base")
image = cv2.imread("path/to/document.jpg")

# Extract text only
text = model.ocr_image(images=image)
print(f"Text: {text[0]}")

# Extract text with bounding boxes
detections = model.parse_document(images=image)
print(detections[0])  # Detections with text in bboxes_metadata
```

### Phrase Grounding

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("florence-2-base")
image = cv2.imread("path/to/image.jpg")

# Find objects matching a phrase
detections = model.ground_phrase(images=image, phrase="red car")
print(detections[0])
```

### Instance Segmentation

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("florence-2-base")
image = cv2.imread("path/to/image.jpg")

# Segment objects matching a phrase
instance_detections = model.segment_phrase(images=image, phrase="person")
print(instance_detections[0])  # InstanceDetections with masks
```

### Using Fine-tuned Models

```python
import cv2
from inference_models import AutoModel

# Load your fine-tuned model from Roboflow
model = AutoModel.from_pretrained(
    "your-workspace/your-model/version",
    api_key="your_roboflow_api_key"
)

image = cv2.imread("path/to/image.jpg")

# Use the model (API depends on what you fine-tuned for)
detections = model.detect_objects(images=image)
print(detections[0])
```

## Workflows Integration

Florence-2 can be used in Roboflow Workflows for complex computer vision pipelines. The Florence-2 block supports all task types and can be combined with other blocks for advanced processing.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - Florence-2 requires GPU for acceptable performance
2. **Choose the right model size** - Use `base` for speed, `large` for accuracy
3. **Optimize prompts** - Use specific task prompts for better results
4. **Fine-tune for your domain** - LoRA fine-tuning significantly improves task-specific performance
5. **Batch processing** - Process multiple images together when possible

## Key Differences from Other VLMs

| Feature | Florence-2 | PaliGemma | Qwen2.5-VL |
|---------|------------|-----------|------------|
| **Model Size** | 0.23B - 0.77B | 3B - 10B | 2B - 72B |
| **Speed** | Fast | Medium | Slower |
| **Task Versatility** | Very High | High | Very High |
| **OCR Quality** | Excellent | Good | Excellent |
| **Fine-tuning** | LoRA | LoRA | LoRA |
| **License** | MIT | Gemma License | Apache 2.0 |
