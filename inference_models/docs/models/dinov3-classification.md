# DINOv3 - Classification

DINOv3 is a state-of-the-art self-supervised vision transformer developed by Meta AI. It learns powerful visual representations without labels, making it excellent for transfer learning and classification tasks.

## Overview

DINOv3 for classification leverages self-supervised learning to create robust feature representations. Key features include:

- **Self-supervised learning** - Trained without labels using advanced techniques
- **Powerful representations** - Excellent transfer learning capabilities
- **Vision transformer backbone** - Based on ViT architecture with improvements
- **Multiple model sizes** - From small to giant variants
- **Strong generalization** - Works well across diverse domains with minimal fine-tuning

## License

**Apache 2.0**

!!! info "Open Source License"
    DINOv3 is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

DINOv3 models must be trained on Roboflow or uploaded as custom weights. There are no pre-trained public model IDs available for classification tasks.

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |

## Roboflow Platform Compatibility

| Feature | Supported |
|---------|-----------|
| **Training** | ✅ Train custom models on Roboflow |
| **Upload Weights** | ✅ Upload pre-trained weights ([guide](https://docs.roboflow.com/deploy/upload-custom-weights)) |
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via Classification block |
| **Edge Deployment (Jetson)** | ✅ Deploy on NVIDIA Jetson devices |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Installation

Install with PyTorch extras:

- **PyTorch**: `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126`

## Usage Example

```python
import cv2
from inference_models import AutoModel

# Load your custom model (requires Roboflow API key)
model = AutoModel.from_pretrained(
    "my-project-abc123/2",
    api_key="your_roboflow_api_key"
)
image = cv2.imread("path/to/image.jpg")

# Run inference
prediction = model(image)

# Get top prediction
top_class_id = prediction.class_id[0].item()
top_class = model.class_names[top_class_id]
confidence = prediction.confidence[0][top_class_id].item()

print(f"Class: {top_class}")
print(f"Confidence: {confidence:.2f}")

# Get all class confidences
for idx, class_name in enumerate(model.class_names):
    conf = prediction.confidence[0][idx].item()
    print(f"{class_name}: {conf:.3f}")
```

## Model Variants

DINOv3 is available in multiple sizes for different performance requirements:

| Variant | Parameters | Use Case |
|---------|------------|----------|
| Small | ~22M | Fast inference, edge deployment |
| Base | ~86M | Balanced performance |
| Large | ~304M | High accuracy applications |
| Giant | ~1.1B | Maximum accuracy, research |

