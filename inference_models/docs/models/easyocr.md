# EasyOCR - Optical Character Recognition

EasyOCR is a versatile OCR solution developed by JaidedAI that supports over 80 languages. It excels at detecting and recognizing text in natural scenes and real-world images.

## Overview

**Resources**: [GitHub Repository](https://github.com/JaidedAI/EasyOCR)

EasyOCR provides robust text detection and recognition with extensive language support. Key features include:

- **80+ languages** - Extensive multi-language support
- **Scene text optimized** - Excellent for real-world images (signs, labels, products)
- **Two-stage pipeline** - Separate detection and recognition for flexibility
- **Easy to use** - Simple API with good defaults
- **Active community** - Well-maintained with regular updates

## OCR Type: Structured OCR

**Structured OCR** detects text regions in an image and recognizes the text within each region, returning both the text content and bounding box coordinates for each detected text block.

### When to Use EasyOCR

- ✅ **Scene text** - Signs, billboards, product labels, street signs
- ✅ **Multi-language text** - Documents with multiple languages
- ✅ **Natural images** - Photos with text in various orientations
- ✅ **Curved or rotated text** - Text that isn't perfectly horizontal
- ✅ **Text localization needed** - When you need to know where text appears

### When to Use Other OCR Models

- **DocTR**: Better for structured documents (invoices, forms, scanned pages)
- **TrOCR**: Better for single-line, pre-cropped text (serial numbers, labels)

## License

**Apache 2.0**

!!! info "Open Source License"
    EasyOCR is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

Pre-trained EasyOCR models are available for various languages via the Roboflow API and **require a Roboflow API key**.

!!! info "Getting a Roboflow API Key"
    To use EasyOCR models, you'll need a [Roboflow account](https://app.roboflow.com/) (free) and [API key](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key).

EasyOCR model IDs combine a recognition model (language) and a detection model using the format: `easy-ocr-{language}/{detection}`

### Recognition Models (Languages)

Recognition models read text in specific languages.

| Language | ID Chunk | Supported Languages |
|----------|----------|---------------------|
| English | `english` | English |
| Latin | `latin` | English, Spanish, French, Italian, Portuguese, German, Polish, Dutch, Latin |
| Simplified Chinese | `zh-sim` | English, Simplified Chinese |
| Japanese | `japanese` | English, Japanese |
| Korean | `korean` | English, Korean |
| Telugu | `telugu` | English, Telugu |
| Kannada | `kannada` | English, Kannada |
| Cyrillic | `cyrillic` | Russian, Ukrainian, Bulgarian, Serbian, and other Cyrillic scripts |

### Detection Models

Detection models locate text regions in the image.

| Model | ID Chunk | Description |
|-------|----------|-------------|
| CRAFT | `craft` | Character Region Awareness for Text detection |
| DBNet18 | `dbnet18` | Differentiable Binarization with ResNet18 backbone |
| DBNet50 | `dbnet50` | Differentiable Binarization with ResNet50 backbone |

### Example Model IDs

Combine any language with any detection model:

- `easy-ocr-english/craft` - English + CRAFT detection
- `easy-ocr-latin/dbnet50` - Latin languages + DBNet50 detection (highest accuracy)
- `easy-ocr-zh-sim/craft` - Simplified Chinese + CRAFT detection
- `easy-ocr-japanese/dbnet18` - Japanese + DBNet18 detection (balanced)

**Total combinations**: 8 recognition models × 3 detection models = 24 possible model configurations

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
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via EasyOCR block |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Installation

Install with PyTorch extras:

- **PyTorch**: `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126`

## Usage Example

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load EasyOCR model for English with CRAFT detection
model = AutoModel.from_pretrained(
    "easy-ocr-english/craft",
    api_key="your_roboflow_api_key"
)

# Load image
image = cv2.imread("path/to/image.jpg")

# Run inference - returns (texts, detections)
texts, detections = model(image)

# Print detected text
for text in texts:
    print(f"Detected: {text}")

# Visualize results with supervision
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Annotate image with bounding boxes and text labels
annotated_image = box_annotator.annotate(scene=image.copy(), detections=detections[0])
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections[0], labels=texts)

# Save or display
cv2.imwrite("output.jpg", annotated_image)
```

## Output Format

EasyOCR returns a tuple of `(List[str], List[Detections])`:

- **texts**: List of recognized text strings, one per detected text region
- **detections**: List of Detections objects with bounding boxes and metadata

This structured output allows you to know both what text was detected and where it appears in the image.

