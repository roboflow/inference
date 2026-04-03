# ViT - Classification

Vision Transformer (ViT) is a transformer-based architecture that applies the transformer model directly to image patches, achieving excellent performance on image classification tasks.

## Overview

ViT for classification brings the power of transformers to computer vision. Key features include:

- **Transformer architecture** - Self-attention mechanisms for global context
- **Patch-based processing** - Images divided into patches and processed as sequences
- **Scalable design** - Performance improves with model size and data
- **Strong transfer learning** - Excellent feature representations for fine-tuning
- **Multiple variants** - Different sizes and patch configurations available

## License

**Apache 2.0**

!!! info "Open Source License"
    Vision Transformer is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

ViT models must be trained on Roboflow or uploaded as custom weights. There are no pre-trained public model IDs available for classification tasks.

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `hugging-face` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |
| `trt` | `trt10` |

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

Install with one of the following extras depending on your backend:

- **ONNX**: `onnx-cpu`, `onnx-cu12`
- **TensorRT**: `trt10` (requires CUDA 12.x)
- **Hugging Face (PyTorch)**: `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126`

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

