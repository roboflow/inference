# SAM2 - Interactive Segmentation

Segment Anything Model 2 (SAM2) is Meta AI's next-generation foundation model for interactive image and video segmentation. It improves upon SAM with better accuracy, speed, and video tracking capabilities.

## Overview

SAM2 provides advanced interactive segmentation with:

- **Improved accuracy** - Better mask quality compared to SAM
- **Faster inference** - Optimized architecture with better speed/quality tradeoff
- **Video support** - Track objects across video frames
- **Image embedding** - Efficient caching for multiple segmentations
- **Multi-mask output** - Generate multiple mask proposals
- **Hiera architecture** - Efficient hierarchical vision transformer

## License

**Apache 2.0**

## Pre-trained Model IDs

All SAM2 models require a **Roboflow API key**.

| Model Size | Model ID |
|------------|----------|
| Tiny | `sam2/hiera_tiny` |
| Small | `sam2/hiera_small` |
| Base+ | `sam2/hiera_b_plus` |
| Large | `sam2/hiera_large` |

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

!!! warning "GPU Recommended"
    Interactive segmentation models work best on GPU. CPU inference will be significantly slower.

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ❌ No custom training |
| **Upload Weights** | ❌ Not applicable |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) |
| **Edge Deployment (Jetson)** | ⚠️ Experimental (may fail on devices with limited VRAM) |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

```python
import cv2
import numpy as np
from inference_models import AutoModel

# Load model (requires API key)
model = AutoModel.from_pretrained("sam2/hiera_b_plus", api_key="your_api_key")
image = cv2.imread("path/to/image.jpg")

# Step 1: Embed the image (optional but recommended for multiple segmentations)
embeddings = model.embed_images(image)

# Step 2: Segment with point prompts
# Positive point (foreground) at (100, 150)
point_coords = np.array([[100, 150]])
point_labels = np.array([1])  # 1 = foreground, 0 = background

results = model.segment_images(
    embeddings=embeddings,
    point_coordinates=point_coords,
    point_labels=point_labels
)

# Access masks and scores
masks = results[0].masks  # Shape: (num_masks, H, W)
scores = results[0].scores  # Confidence scores for each mask
logits = results[0].logits  # Low-resolution logits for refinement

# Use the best mask (highest score)
best_mask_idx = scores.argmax()
best_mask = masks[best_mask_idx]
```

## Output Format

Returns: `List[SAM2Prediction]` (one per image in batch)

Each `SAM2Prediction` contains:
- **masks** - Binary masks (torch.Tensor)
- **scores** - Confidence scores for each mask
- **logits** - Low-resolution logits for mask refinement

## Prompting Options

**Point Prompts:**
```python
# Positive points (foreground)
point_coords = np.array([[x1, y1], [x2, y2]])
point_labels = np.array([1, 1])

# Negative points (background)
point_coords = np.array([[x1, y1]])
point_labels = np.array([0])
```

**Box Prompts:**
```python
# Box format: [x_min, y_min, x_max, y_max]
boxes = np.array([[50, 50, 200, 200]])
results = model.segment_images(embeddings=embeddings, boxes=boxes)
```

**Mask Refinement:**
```python
# Use previous mask logits to refine segmentation
results = model.segment_images(
    embeddings=embeddings,
    point_coordinates=point_coords,
    point_labels=point_labels,
    mask_input=previous_logits
)
```

## Batch Processing

```python
# Process multiple images efficiently
images = [cv2.imread(f"image_{i}.jpg") for i in range(4)]
embeddings = model.embed_images(images)

# Segment all images with the same prompt
point_coords = [np.array([[100, 150]])] * 4
point_labels = [np.array([1])] * 4

results = model.segment_images(
    embeddings=embeddings,
    point_coordinates=point_coords,
    point_labels=point_labels
)
```

## Performance Optimization with Caching

SAM2 supports two types of caching for significant performance improvements:

### 1. Image Embeddings Cache

The **most important optimization** - caches the compute-heavy image encoding step. When you need to segment the same image multiple times with different prompts, embeddings are computed only once.

```python
from inference_models import AutoModel
from inference_models.models.sam2.cache import Sam2ImageEmbeddingsInMemoryCache
import cv2
import numpy as np

# Create cache with size limit (number of images to cache)
embeddings_cache = Sam2ImageEmbeddingsInMemoryCache.init(
    size_limit=100,  # Cache up to 100 image embeddings
    send_to_cpu=True  # Move cached embeddings to CPU to save GPU memory
)

# Load model with cache
model = AutoModel.from_pretrained(
    "sam2/hiera_b_plus",
    api_key="your_api_key",
    sam2_image_embeddings_cache=embeddings_cache
)

image = cv2.imread("image.jpg")

# First call: computes embeddings (slow)
embeddings = model.embed_images(image, use_embeddings_cache=True)

# Refine with different prompts - embeddings reused from cache (fast!)
for point in [(100, 150), (200, 250), (300, 350)]:
    results = model.segment_images(
        embeddings=embeddings,
        point_coordinates=np.array([[point[0], point[1]]]),
        point_labels=np.array([1])
    )
    # Process results...
```

### 2. Low-Resolution Masks Cache

Caches the low-resolution mask logits from previous segmentations for iterative refinement. SAM2's cache is more sophisticated than SAM's, storing multiple mask variants per image.

```python
from inference_models.models.sam2.cache import Sam2LowResolutionMasksInMemoryCache

# Create mask cache
masks_cache = Sam2LowResolutionMasksInMemoryCache.init(
    size_limit=500,  # Cache up to 500 mask logits
    send_to_cpu=True
)

# Load model with both caches
model = AutoModel.from_pretrained(
    "sam2/hiera_b_plus",
    api_key="your_api_key",
    sam2_image_embeddings_cache=embeddings_cache,
    sam2_low_resolution_masks_cache=masks_cache
)

# First segmentation - mask logits cached automatically
results = model.segment_images(
    image,
    point_coordinates=np.array([[100, 150]]),
    point_labels=np.array([1]),
    use_mask_input_cache=True
)

# Refinement - uses cached mask logits as input
refined_results = model.segment_images(
    image,
    point_coordinates=np.array([[100, 150], [120, 160]]),  # Add more points
    point_labels=np.array([1, 1]),
    use_mask_input_cache=True  # Automatically uses cached logits
)
```

### Cache Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `size_limit` | Maximum number of entries to cache | Required |
| `send_to_cpu` | Move cached data to CPU to save GPU memory | `True` |
| `use_embeddings_cache` | Enable embeddings cache lookup/save | `True` |
| `use_mask_input_cache` | Enable mask logits cache lookup/save | `True` |

!!! tip "Performance Impact"
    - **Without cache**: Each segmentation requires full image encoding (compute-heavy)
    - **With embeddings cache**: Subsequent segmentations on same image are significantly faster
    - **Major speedup** for interactive annotation workflows where you refine prompts on the same image
    - SAM2 is generally faster than SAM for the same quality level

## Use Cases

SAM2 is ideal for:

- ✅ **Interactive annotation** - Quickly segment objects with minimal user input
- ✅ **Data labeling** - Accelerate dataset creation with point/box prompts
- ✅ **Video object tracking** - Track and segment objects across video frames
- ✅ **Object isolation** - Extract specific objects from images
- ✅ **Mask refinement** - Iteratively improve segmentation quality
- ✅ **Zero-shot segmentation** - Segment novel objects without training

