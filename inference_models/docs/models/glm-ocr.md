# GLM-OCR - Vision-Language OCR

GLM-OCR is a vision-language model developed by Zhipu AI (ZAI) that excels at optical character recognition using an image-text-to-text architecture. It combines visual understanding with text generation for accurate text recognition.

## Overview

**Resources**: [Hugging Face Model](https://huggingface.co/zai-org/GLM-OCR)

GLM-OCR provides high-quality text recognition using a modern vision-language approach. Key features include:

- **Vision-language architecture** - Uses AutoModelForImageTextToText for end-to-end OCR
- **Prompt-based** - Customizable prompts for different recognition tasks
- **High accuracy** - Strong performance on diverse text types
- **Flash Attention support** - Automatic flash attention on supported GPUs (Ampere+)

## OCR Type: Unstructured OCR

**Unstructured OCR** recognizes text from an image using a vision-language model. It returns the recognized text string based on the given prompt.

### When to Use GLM-OCR

- **Serial numbers & labels** - Recognizing text from product labels, serial numbers
- **Scene text** - Text in natural images and photos
- **Document text** - Recognizing text from document images
- **Custom prompts** - When you need to guide the model with specific instructions

### When to Use Other OCR Models

- **DocTR**: Better for full document layout analysis with text detection and bounding boxes
- **EasyOCR**: Better for multi-language scene text with detection and localization
- **TrOCR**: Lighter alternative for simple single-line pre-cropped text

## License

**MIT License**

!!! info "Open Source License"
    GLM-OCR is licensed under MIT by Zhipu AI (ZAI), making it free for both commercial and non-commercial use without restrictions.

    Learn more: [MIT License](https://opensource.org/licenses/MIT)

## Pre-trained Model IDs

Pre-trained GLM-OCR models are available via the Roboflow API and **require a Roboflow API key**.

!!! info "Getting a Roboflow API Key"
    To use GLM-OCR models, you'll need a [Roboflow account](https://app.roboflow.com/) (free) and [API key](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key).

| Model ID | Description |
|----------|-------------|
| `glm-ocr` | GLM-OCR vision-language model for text recognition |

!!! warning "GPU Required"
    GLM-OCR uses bfloat16 precision and requires GPU acceleration. CPU inference is not supported.

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not available for training |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via LMM block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU required) |

## Installation

Install with PyTorch GPU extras:

- **PyTorch**: `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126`

## Usage Examples

### Basic Text Recognition

```python
import cv2
from inference_models import AutoModel

# Load GLM-OCR model
model = AutoModel.from_pretrained(
    "glm-ocr",
    api_key="your_roboflow_api_key"
)

# Load image
image = cv2.imread("path/to/image.jpg")

# Run OCR with default prompt ("Text Recognition:")
results = model.prompt(images=image)
print(f"Recognized text: {results[0]}")
```

### Custom Prompt

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained(
    "glm-ocr",
    api_key="your_roboflow_api_key"
)

image = cv2.imread("path/to/serial_number.png")

# Use a custom prompt to guide recognition
results = model.prompt(
    images=image,
    prompt="Read the serial number in this image:",
    max_new_tokens=100
)
print(f"Serial number: {results[0]}")
```

### Inference Server (HTTP API)

GLM-OCR can also be used via the inference server HTTP API:

```python
import requests

response = requests.post(
    "http://localhost:9001/infer/lmm",
    json={
        "api_key": "your_roboflow_api_key",
        "model_id": "glm-ocr",
        "image": {
            "type": "url",
            "value": "https://example.com/image.png",
        },
        "prompt": "Text Recognition:",
    },
)

data = response.json()
print(f"Recognized text: {data['response']}")
```

## Output Format

GLM-OCR returns a `List[str]` containing the recognized text from the input images.

- **Single image input**: Returns a list with one string
- The default prompt is `"Text Recognition:"` if none is provided

## Performance Tips

1. **Use GPU** - GLM-OCR requires GPU with bfloat16 support
2. **Flash Attention** - Automatically enabled on Ampere+ GPUs (compute capability >= 8.0) for faster inference
3. **Adjust max_new_tokens** - Increase for longer text passages, decrease for short labels
4. **Specific prompts** - Use targeted prompts like "Read the serial number:" for better results on specific tasks
