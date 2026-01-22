# DocTR - Optical Character Recognition

DocTR (Document Text Recognition) is a comprehensive OCR solution developed by Mindee. It combines text detection and recognition to extract text from documents and images.

## Overview

**Resources**: [GitHub Repository](https://github.com/mindee/doctr) | [Documentation](https://mindee.github.io/doctr/)

DocTR provides end-to-end document text recognition with both detection and recognition stages. Key features include:

- **Two-stage pipeline** - Separate detection and recognition models for optimal performance
- **Document-focused** - Optimized for document layouts and structured text
- **Multiple architectures** - Various detection and recognition model combinations
- **Production-ready** - Battle-tested on real-world documents
- **Flexible deployment** - Multiple model size options for different use cases

## OCR Type: Structured OCR

**Structured OCR** detects text regions in an image and recognizes the text within each region, returning both the text content and bounding box coordinates for each detected text block.

### When to Use DocTR

- ✅ **Documents and forms** - Scanned documents, invoices, receipts
- ✅ **Multi-line text** - Paragraphs and structured layouts
- ✅ **Text localization needed** - When you need to know where text appears
- ✅ **Mixed content** - Documents with text in various locations

### When to Use Other OCR Models

- **EasyOCR**: Better for scene text (signs, labels) and multi-language support
- **TrOCR**: Better for single-line, pre-cropped text (serial numbers, labels)

## License

**Apache 2.0**

!!! info "Open Source License"
    DocTR is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

Pre-trained DocTR models are available via the Roboflow API and **require a Roboflow API key**.

!!! info "Getting a Roboflow API Key"
    To use DocTR models, you'll need a [Roboflow account](https://app.roboflow.com/) (free) and [API key](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key).

DocTR model IDs combine a detection model and a recognition model using the format: `doctr-{detection}/{recognition}`

### Detection Models

Detection models locate text regions in the image and output bounding boxes.

| Model | ID Chunk | Speed | Accuracy | Description |
|-------|----------|-------|----------|-------------|
| FAST Tiny | `fast-t` | Very Fast | Low | Lightweight FAST architecture |
| FAST Small | `fast-s` | Fast | Medium | Balanced FAST variant |
| FAST Base | `fast-b` | Medium | Good | Standard FAST model |
| DB ResNet50 | `dbnet-rn50` | Medium | High | Differentiable Binarization with ResNet50 backbone |
| DB ResNet34 | `dbnet-rn34` | Medium | High | DB with ResNet34 backbone |
| DB MobileNet V3 Large | `db-net-mobilenet-v3-l` | Fast | Medium | DB with efficient MobileNet backbone |
| LinkNet ResNet18 | `linknet-rn18` | Fast | Medium | LinkNet segmentation with ResNet18 |
| LinkNet ResNet34 | `linknet-rn34` | Medium | Good | LinkNet with ResNet34 |
| LinkNet ResNet50 | `linknet-rn50` | Medium | High | LinkNet with ResNet50 |

### Recognition Models

Recognition models read the text content from detected regions.

| Model | ID Chunk | Speed | Accuracy | Description |
|-------|----------|-------|----------|-------------|
| CRNN VGG16 | `crnn-vgg16` | Medium | Good | CNN-RNN hybrid with VGG16 encoder |
| CRNN MobileNet V3 Small | `crnn-mobilenet-v3-small` | Very Fast | Medium | Efficient CRNN with small MobileNet |
| CRNN MobileNet V3 Large | `crnn-mobilenet-v3-large` | Fast | Good | CRNN with larger MobileNet |
| SAR ResNet31 | `sar-rn31` | Slow | High | Show, Attend and Read - attention-based |
| MASTER | `master` | Slow | High | Multi-Aspect Non-local Network |
| ViTSTR Small | `vitstr-s` | Medium | Good | Vision Transformer for text recognition |
| PARSeq | `parseq` | Medium | High | Permutation Language Modeling - state-of-the-art |

### Example Model IDs

Combine any detection model with any recognition model:

- `doctr-dbnet-rn50/crnn-vgg16` - DB ResNet50 detection + CRNN VGG16 recognition
- `doctr-fast-b/parseq` - FAST Base detection + PARSeq recognition
- `doctr-db-net-mobilenet-v3-l/crnn-mobilenet-v3-small` - MobileNet detection + MobileNet recognition (fastest)
- `doctr-dbnet-rn50/sar-rn31` - DB ResNet50 detection + SAR ResNet31 recognition (highest accuracy)

**Total combinations**: 9 detection models × 7 recognition models = 63 possible model configurations

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
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via OCR block |
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

# Load DocTR model (DB ResNet50 detection + PARSeq recognition)
model = AutoModel.from_pretrained(
    "doctr-dbnet-rn50/parseq",
    api_key="your_roboflow_api_key"
)

# Load image
image = cv2.imread("path/to/document.jpg")

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

DocTR returns a tuple of `(List[str], List[Detections])`:

- **texts**: List of recognized text strings, one per detected text region
- **detections**: List of Detections objects with bounding boxes and metadata

This structured output allows you to know both what text was detected and where it appears in the image.

