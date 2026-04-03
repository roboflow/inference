# TrOCR - Transformer-based Optical Character Recognition

TrOCR is a transformer-based OCR model developed by Microsoft that excels at recognizing text from pre-cropped image regions. It uses a vision encoder-decoder architecture for end-to-end text recognition.

## Overview

**Resources**: [Research Paper](https://arxiv.org/abs/2109.10282) | [Hugging Face Models](https://huggingface.co/models?search=microsoft/trocr)

TrOCR provides state-of-the-art text recognition for single-line text regions. Key features include:

- **Transformer architecture** - Modern encoder-decoder design for superior accuracy
- **Pre-cropped text** - Optimized for single text regions (no detection stage)
- **High accuracy** - Excellent recognition quality on clean text
- **Multiple variants** - Small, base, and large models for different accuracy/speed tradeoffs
- **Handwriting support** - Specialized models for handwritten text (can be added upon request)

## OCR Type: Unstructured OCR

**Unstructured OCR** recognizes text from a pre-cropped image region containing a single line of text. It returns only the recognized text string without bounding box information.

### When to Use TrOCR

- ✅ **Pre-cropped text** - When you already have isolated text regions
- ✅ **Single-line text** - Serial numbers, labels, captions, single words
- ✅ **High accuracy needed** - When recognition quality is critical
- ✅ **Handwritten text** - Use handwriting-specific models
- ✅ **After object detection** - Recognize text in detected bounding boxes

### When to Use Other OCR Models

- **DocTR**: Better for full documents where you need to detect text locations
- **EasyOCR**: Better for scene text with detection and multi-language support

## License

**MIT License**

!!! info "Open Source License"
    TrOCR is licensed under MIT, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [MIT License](https://opensource.org/licenses/MIT)

## Pre-trained Model IDs

Pre-trained TrOCR models are available via the Roboflow API and **require a Roboflow API key**.

!!! info "Getting a Roboflow API Key"
    To use TrOCR models, you'll need a [Roboflow account](https://app.roboflow.com/) (free) and [API key](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key).

| Model Variant | Model ID | Use Case | Size |
|---------------|----------|----------|------|
| Small (Printed) | `microsoft/trocr-small-printed` | Fast printed text recognition | Small |
| Base (Printed) | `microsoft/trocr-base-printed` | Balanced printed text recognition | Base |
| Large (Printed) | `microsoft/trocr-large-printed` | High-accuracy printed text | Large |

**Recommendation**: Start with small or base models for faster inference. Use large models when accuracy is critical.

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
from inference_models import AutoModel

# Load TrOCR model for printed text
model = AutoModel.from_pretrained(
    "microsoft/trocr-base-printed",
    api_key="your_roboflow_api_key"
)

# Load pre-cropped images containing single lines of text
images = [
    cv2.imread("path/to/cropped_text1.jpg"),
    cv2.imread("path/to/cropped_text2.jpg"),
]

# Run inference - returns list of recognized text strings
texts = model(images)

# Print results
for i, text in enumerate(texts):
    print(f"Image {i+1}: {text}")
```

### Combining with Object Detection

TrOCR works great for recognizing text in detected regions (e.g., after object detection). Instead of manually combining models in code, we recommend using **Roboflow Workflows** to easily create pipelines that:

- Detect text regions with object detection or DocTR/EasyOCR
- Crop detected regions
- Run TrOCR on each cropped region
- Return structured results

Learn more: [Roboflow Workflows Documentation](https://docs.roboflow.com/workflows)

## Output Format

TrOCR returns a `List[str]` containing the recognized text from the input images.

- **Single image input**: Returns a list with one string
- **Batch input**: Returns a list of strings, one per image

**Important**: Each input image should contain a single line of text. For multi-line text or text detection, use DocTR or EasyOCR instead.
