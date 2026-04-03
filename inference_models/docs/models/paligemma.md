# PaliGemma / PaliGemma2 - Vision Language Model

PaliGemma and PaliGemma2 are versatile vision-language models developed by Google Research that combine the SigLIP vision encoder with the Gemma language model for multimodal understanding.

## Overview

PaliGemma is a powerful VLM capable of handling diverse vision-language tasks:

- **Visual Question Answering** - Answer questions about image content
- **Image Captioning** - Generate descriptive captions for images
- **Object Detection** - Detect and locate objects through text prompts
- **OCR** - Extract and recognize text from images
- **Document Understanding** - Parse and analyze document content

!!! warning "GPU Recommended"
    PaliGemma works best with GPU acceleration. CPU inference may be very slow or may not work properly.

!!! info "License & Attribution"
    **License**: Gemma Terms of Use<br>**Source**: [Google Research](https://github.com/google-research/big_vision)<br>**Paper**: [PaliGemma: A versatile 3B VLM for transfer](https://arxiv.org/abs/2407.07726)<br>**Terms**: By using PaliGemma you agree to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms)

## Pre-trained Model IDs

PaliGemma and PaliGemma2 pre-trained models are available and do **not** require a Roboflow API key.

### PaliGemma Models

| Model ID | Description |
|----------|-------------|
| `paligemma-3b-mix-224` | 3B model with 224x224 resolution - general purpose |
| `paligemma-3b-mix-448` | 3B model with 448x448 resolution - higher quality |
| `paligemma-3b-ft-cococap-224` | Fine-tuned for image captioning (224px) |
| `paligemma-3b-ft-cococap-448` | Fine-tuned for image captioning (448px) |
| `paligemma-3b-ft-vqav2-224` | Fine-tuned for visual question answering (224px) |
| `paligemma-3b-ft-vqav2-448` | Fine-tuned for visual question answering (448px) |
| `paligemma-3b-ft-docvqa-224` | Fine-tuned for document VQA (224px) |
| `paligemma-3b-ft-docvqa-448` | Fine-tuned for document VQA (448px) |
| `paligemma-3b-ft-ocrvqa-224` | Fine-tuned for OCR VQA (224px) |
| `paligemma-3b-ft-ocrvqa-448` | Fine-tuned for OCR VQA (448px) |
| `paligemma-3b-ft-screen2words-224` | Fine-tuned for UI understanding (224px) |
| `paligemma-3b-ft-screen2words-448` | Fine-tuned for UI understanding (448px) |
| `paligemma-3b-ft-tallyqa-224` | Fine-tuned for counting tasks (224px) |
| `paligemma-3b-ft-tallyqa-448` | Fine-tuned for counting tasks (448px) |

### PaliGemma2 Models

| Model ID | Description |
|----------|-------------|
| `paligemma2-3b-pt-224` | PaliGemma2 3B pre-trained model with 224x224 resolution |

You can also use fine-tuned models from Roboflow by specifying `project/version` as the model ID (requires API key).

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ LoRA fine-tuning only ([Guide](https://blog.roboflow.com/paligemma-multimodal-vision/)) |
| **Upload Weights** | ✅ Upload fine-tuned models |
| **Serverless API (v2)** | ⚠️ Limited support (not yet fully stable) |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via PaliGemma block |
| **Edge Deployment (Jetson)** | ❌ Not supported |
| **Self-Hosting** | ✅ Deploy with `inference-models` (GPU recommended) |

## Training & Fine-tuning

PaliGemma supports **LoRA (Low-Rank Adaptation) fine-tuning only** on the Roboflow platform. This allows you to adapt the model to your specific use case without training the entire model.

### When to Fine-tune PaliGemma

Fine-tuning PaliGemma is beneficial when you need:

- **Domain-specific VQA** - Answer questions specific to your industry or use case
- **Custom captioning** - Generate captions with domain-specific terminology
- **Specialized document understanding** - Parse forms, receipts, or technical documents
- **Task-specific performance** - Optimize for particular vision-language tasks

### Recommended Use Cases for Fine-tuning

- ✅ **Medical imaging** - Answer questions about medical scans or reports
- ✅ **Document processing** - Extract information from invoices, forms, or contracts
- ✅ **E-commerce** - Describe products or answer product-related questions
- ✅ **Education** - Answer questions about diagrams, charts, or educational content
- ✅ **Accessibility** - Generate detailed descriptions for visually impaired users

Learn more: [PaliGemma Multimodal Vision Guide](https://blog.roboflow.com/paligemma-multimodal-vision/)

## Supported Tasks

PaliGemma supports multiple vision-language tasks through natural language prompts:

| Task | When to Use |
|------|-------------|
| Visual Question Answering | Answer any question about image content - most versatile task for general queries |
| Image Captioning | Generate descriptive captions by using prompts like "caption" or "describe this image" |
| Object Detection | Detect objects by asking "detect [object]" or similar prompts |
| OCR | Extract text by using prompts like "read the text" or "what does it say" |
| Document VQA | Ask questions about document content, forms, or structured data |
| Counting | Count objects by asking "how many [objects]" |

## Usage Examples

### Visual Question Answering

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("paligemma-3b-mix-448")
image = cv2.imread("path/to/image.jpg")

# Ask a question
answers = model.prompt(images=image, prompt="What is in this image?")
print(f"Answer: {answers[0]}")
```

### Image Captioning

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("paligemma-3b-ft-cococap-448")
image = cv2.imread("path/to/image.jpg")

# Generate caption
captions = model.prompt(images=image, prompt="caption")
print(f"Caption: {captions[0]}")
```

### Document Understanding

```python
import cv2
from inference_models import AutoModel

# Load model fine-tuned for document VQA
model = AutoModel.from_pretrained("paligemma-3b-ft-docvqa-448")
image = cv2.imread("path/to/document.jpg")

# Ask about document content
answers = model.prompt(images=image, prompt="What is the total amount?")
print(f"Answer: {answers[0]}")
```

### Using Fine-tuned Models

```python
import cv2
from inference_models import AutoModel

# Load your fine-tuned model from Roboflow
model = AutoModel.from_pretrained(
    "your-project/version",
    api_key="your_roboflow_api_key"
)

image = cv2.imread("path/to/image.jpg")

# Use with custom prompt for your use case
answers = model.prompt(images=image, prompt="your custom question")
print(f"Answer: {answers[0]}")
```

## Workflows Integration

PaliGemma can be used in Roboflow Workflows for complex computer vision pipelines. The PaliGemma block supports all task types and can be combined with other blocks for advanced processing.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Use GPU** - PaliGemma requires GPU for acceptable performance
2. **Choose the right resolution** - Use 224px for speed, 448px for accuracy
3. **Use task-specific models** - Fine-tuned models (e.g., `ft-docvqa`) perform better on specific tasks
4. **Optimize prompts** - Clear, specific prompts yield better results
5. **Fine-tune for your domain** - LoRA fine-tuning significantly improves task-specific performance

