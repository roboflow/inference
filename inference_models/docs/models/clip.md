# CLIP - Embeddings Model

CLIP (Contrastive Language-Image Pre-Training) is a neural network trained on image-text pairs that can generate embeddings for both images and text in a shared vector space, enabling powerful zero-shot classification and similarity search.

## Overview

CLIP is a versatile embeddings model with multiple capabilities:

- **Image Embeddings** - Generate vector representations of images
- **Text Embeddings** - Generate vector representations of text
- **Similarity Comparison** - Compare images and text in a shared embedding space
- **Zero-shot Classification** - Classify images without task-specific training
- **Semantic Search** - Find images based on text descriptions

!!! info "License & Attribution"
    **License**: MIT<br>**Source**: [OpenAI CLIP](https://github.com/openai/CLIP)<br>**Paper**: [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)

## Pre-trained Model IDs

CLIP pre-trained models are available and do **not** require a Roboflow API key.

| Model ID | Description |
|----------|-------------|
| `clip/RN50` | ResNet-50 backbone - fast inference |
| `clip/RN101` | ResNet-101 backbone - better accuracy |
| `clip/RN50x4` | ResNet-50 4x - higher capacity |
| `clip/RN50x16` | ResNet-50 16x - very high capacity |
| `clip/RN50x64` | ResNet-50 64x - highest capacity ResNet |
| `clip/ViT-B-16` | Vision Transformer Base with 16x16 patches |
| `clip/ViT-B-32` | Vision Transformer Base with 32x32 patches - balanced performance |
| `clip/ViT-L-14` | Vision Transformer Large with 14x14 patches - high accuracy |
| `clip/ViT-L-14-336px` | Vision Transformer Large with 336px input - highest accuracy |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |
| `onnx` | `onnx-cpu`, `onnx-gpu` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not supported |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ✅ Available for embeddings and comparison |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) via CLIP block |
| **Edge Deployment (Jetson)** | ✅ Supported with appropriate backend |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Supported Tasks

CLIP supports multiple embedding and comparison tasks:

| Task | Method | When to Use |
|------|--------|-------------|
| Image Embeddings | `embed_image()` | Generate vector representations of images for similarity search or clustering |
| Text Embeddings | `embed_text()` | Generate vector representations of text descriptions |
| Similarity Comparison | `compare()` | Compare images with text or other images to find semantic similarity |

## Usage Examples

### Image Embeddings

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("clip/ViT-B-32")
image = cv2.imread("path/to/image.jpg")

# Generate image embedding
embedding = model.embed_image(image)
print(f"Embedding shape: {embedding.shape}")
```

### Text Embeddings

```python
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("clip/ViT-B-32")

# Generate text embedding
text_embedding = model.embed_text("a photo of a cat")
print(f"Text embedding shape: {text_embedding.shape}")
```

### Image-Text Similarity

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("clip/ViT-B-32")
image = cv2.imread("path/to/image.jpg")

# Compare image with text descriptions
similarity = model.compare(
    subject=image,
    prompt=["a photo of a cat", "a photo of a dog", "a photo of a bird"],
    subject_type="image",
    prompt_type="text"
)
print(f"Similarities: {similarity}")
```

### Zero-shot Classification

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("clip/ViT-B-32")
image = cv2.imread("path/to/image.jpg")

# Classify image with text prompts
classes = ["cat", "dog", "bird", "car", "tree"]
prompts = [f"a photo of a {cls}" for cls in classes]

similarities = model.compare(
    subject=image,
    prompt=prompts,
    subject_type="image",
    prompt_type="text"
)

# Get the most similar class
best_match_idx = similarities.index(max(similarities))
print(f"Predicted class: {classes[best_match_idx]}")
print(f"Confidence: {similarities[best_match_idx]:.2f}")
```

### Batch Processing

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("clip/ViT-B-32")

# Load multiple images
images = [cv2.imread(f"path/to/image{i}.jpg") for i in range(5)]

# Generate embeddings for all images
embeddings = model.embed_images(images)
print(f"Batch embeddings shape: {embeddings.shape}")
```

## Workflows Integration

CLIP can be used in Roboflow Workflows for complex computer vision pipelines, including zero-shot classification and semantic search.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Choose the right model** - ViT models generally have better accuracy, ResNet models are faster
2. **Use ViT-B-32 for balance** - Good trade-off between speed and accuracy
3. **Batch processing** - Process multiple images together for better throughput
4. **Use ONNX backend** - Often faster than PyTorch for inference
5. **Normalize embeddings** - Use cosine similarity for comparing embeddings

