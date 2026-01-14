# Perception Encoder - Embeddings Model

Perception Encoder is Meta's FAIR (Fundamental AI Research) vision encoder that generates high-quality image embeddings for various computer vision tasks including similarity search, clustering, and retrieval.

## Overview

Perception Encoder is a state-of-the-art vision-language model designed for:

- **Image Embeddings** - Generate high-quality vector representations of images
- **Text Embeddings** - Generate vector representations of text descriptions
- **Image-Text Similarity** - Compare images and text in a shared embedding space
- **Semantic Search** - Find visually similar images or match images to text queries
- **Image Clustering** - Group images by visual similarity
- **Feature Extraction** - Extract rich visual features for downstream tasks

!!! warning "License Restrictions"
    **License**: FAIR Noncommercial Research License<br>**Restrictions**: Noncommercial research use only<br>**Source**: [Meta FAIR Perception Models](https://github.com/facebookresearch/perception_models)<br>**Code License**: Apache 2.0

!!! important "Commercial Use"
    Perception Encoder is licensed for **noncommercial research use only**. Commercial applications are not permitted under the FAIR Noncommercial Research License. For commercial use cases, consider using [CLIP](clip.md) instead.

## Pre-trained Model IDs

Perception Encoder pre-trained models are available and do **not** require a Roboflow API key.

| Model ID | Description |
|----------|-------------|
| `perception-encoder/PE-Core-B16-224` | Base model with 16x16 patches, 224px input - balanced performance |
| `perception-encoder/PE-Core-G14-448` | Giant model with 14x14 patches, 448px input - high accuracy |
| `perception-encoder/PE-Core-L14-336` | Large model with 14x14 patches, 336px input - very high accuracy |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ Not supported |
| **Upload Weights** | ❌ Not supported |
| **Serverless API (v2)** | ✅ Available for embeddings |
| **Workflows** | ✅ Use in [Workflows](https://inference.roboflow.com/workflows/about/) |
| **Edge Deployment (Jetson)** | ✅ Supported with PyTorch backend |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Examples

### Image Embeddings

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("perception-encoder/PE-Core-B16-224")
image = cv2.imread("path/to/image.jpg")

# Generate image embedding
embedding = model.embed_image(image)
print(f"Embedding shape: {embedding.shape}")
```

### Text Embeddings

```python
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("perception-encoder/PE-Core-B16-224")

# Generate text embedding
text_embedding = model.embed_text("a photo of a cat")
print(f"Text embedding shape: {text_embedding.shape}")

# Generate embeddings for multiple texts
texts = ["a photo of a cat", "a photo of a dog", "a photo of a bird"]
text_embeddings = model.embed_text(texts)
print(f"Batch text embeddings shape: {text_embeddings.shape}")
```

### Image-Text Similarity

```python
import cv2
import numpy as np
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("perception-encoder/PE-Core-B16-224")
image = cv2.imread("path/to/image.jpg")

# Generate image and text embeddings
image_embedding = model.embed_image(image)
text_embeddings = model.embed_text([
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird"
])

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(image_embedding[0], text_emb) for text_emb in text_embeddings]
print(f"Similarities: {similarities}")

# Find best match
best_match_idx = np.argmax(similarities)
classes = ["cat", "dog", "bird"]
print(f"Best match: {classes[best_match_idx]} (similarity: {similarities[best_match_idx]:.3f})")
```

### Similarity Search

```python
import cv2
import numpy as np
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("perception-encoder/PE-Core-B16-224")

# Load query image and database images
query_image = cv2.imread("path/to/query.jpg")
database_images = [cv2.imread(f"path/to/image{i}.jpg") for i in range(100)]

# Generate embeddings
query_embedding = model.embed_image(query_image)
database_embeddings = model.embed_images(database_images)

# Compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

similarities = [cosine_similarity(query_embedding, db_emb) for db_emb in database_embeddings]

# Find most similar images
top_k = 5
top_indices = np.argsort(similarities)[-top_k:][::-1]
print(f"Top {top_k} most similar images: {top_indices}")
```

### Image Clustering

```python
import cv2
import numpy as np
from sklearn.cluster import KMeans
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("perception-encoder/PE-Core-B16-224")

# Load images
images = [cv2.imread(f"path/to/image{i}.jpg") for i in range(100)]

# Generate embeddings
embeddings = model.embed_images(images)

# Cluster embeddings
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(embeddings)

print(f"Cluster assignments: {cluster_labels}")
```

### Batch Processing

```python
import cv2
from inference_models import AutoModel

# Load model
model = AutoModel.from_pretrained("perception-encoder/PE-Core-L14-336")

# Load multiple images
images = [cv2.imread(f"path/to/image{i}.jpg") for i in range(10)]

# Generate embeddings for all images in batch
embeddings = model.embed_images(images)
print(f"Batch embeddings shape: {embeddings.shape}")
```

## Workflows Integration

Perception Encoder can be used in Roboflow Workflows for complex computer vision pipelines involving image embeddings and similarity search.

Learn more: [Workflows Documentation](https://inference.roboflow.com/workflows/about/)

## Performance Tips

1. **Choose the right model size** - PE-Core-B16-224 for speed, PE-Core-L14-336 or PE-Core-G14-448 for accuracy
2. **Batch processing** - Process multiple images together for better throughput
3. **GPU acceleration** - Use CUDA-enabled PyTorch for faster inference
4. **Normalize embeddings** - Use cosine similarity for comparing embeddings
5. **Cache embeddings** - Pre-compute and store embeddings for large image databases

## Model Comparison

| Model | Input Size | Parameters | Speed | Accuracy |
|-------|-----------|------------|-------|----------|
| PE-Core-B16-224 | 224x224 | Base | Fast | Good |
| PE-Core-L14-336 | 336x336 | Large | Medium | Very Good |
| PE-Core-G14-448 | 448x448 | Giant | Slow | Excellent |

## When to Use Perception Encoder vs CLIP

| Use Case | Recommended Model |
|----------|-------------------|
| **Noncommercial research** | Perception Encoder |
| **Commercial applications** | [CLIP](clip.md) |
| **Image-text embeddings (research)** | Perception Encoder or [CLIP](clip.md) |
| **Zero-shot classification** | Perception Encoder or [CLIP](clip.md) |
| **Highest quality embeddings (research)** | Perception Encoder |
| **MIT licensed model** | [CLIP](clip.md) |

## License Compliance

!!! warning "Important License Information"
    - **Noncommercial use only** - Cannot be used for commercial advantage or monetary compensation
    - **Research purposes** - Intended for research, development, education, processing, or analysis
    - **Attribution required** - Must acknowledge use in publications
    - **Redistribution** - Can only redistribute under the same FAIR Noncommercial Research License
    - **No warranty** - Provided "as is" without warranties
    
    For full license details, see the [FAIR Noncommercial Research License](https://github.com/facebookresearch/perception_models/blob/main/LICENSE.PLM).

