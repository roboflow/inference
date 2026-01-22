# ResNet - Classification

ResNet (Residual Network) is a foundational deep learning architecture that introduced skip connections to enable training of very deep neural networks. It remains one of the most widely used architectures for image classification.

## Overview

ResNet for classification provides robust and reliable image classification performance. Key features include:

- **Residual connections** - Skip connections that enable training of very deep networks
- **Proven architecture** - Battle-tested across countless applications
- **Multiple depths** - From 18 to 152 layers for different accuracy/speed tradeoffs
- **Transfer learning** - Excellent feature extraction for custom datasets
- **Wide adoption** - Extensive community support and resources

## License

**Apache 2.0**

!!! info "Open Source License"
    ResNet is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Pre-trained Model IDs

Pre-trained ResNet models are available with ImageNet weights and are **open access** (no API key required).

| Model | Model ID | Layers | Parameters |
|-------|----------|--------|------------|
| ResNet-18 | `resnet18` | 18 | ~11M |
| ResNet-34 | `resnet34` | 34 | ~21M |
| ResNet-50 | `resnet50` | 50 | ~25M |
| ResNet-101 | `resnet101` | 101 | ~44M |

**Custom model ID format:** `project-url/version` (e.g., `my-project-abc123/2`)

## Supported Backends

| Backend | Extras Required |
|---------|----------------|
| `onnx` | `onnx-cpu`, `onnx-cu12`, `onnx-cu118`, `onnx-jp6-cu126` |
| `torch` | `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` |
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
- **PyTorch**: `torch-cpu`, `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126`

## Usage Example

```python
import cv2
from inference_models import AutoModel

# Load pre-trained model (ImageNet weights)
model = AutoModel.from_pretrained("resnet50")
image = cv2.imread("path/to/image.jpg")

# Run inference
prediction = model(image)

# Get top prediction
top_class_id = prediction.class_id[0].item()
top_class = model.class_names[top_class_id]
confidence = prediction.confidence[0][top_class_id].item()

print(f"Class: {top_class}")
print(f"Confidence: {confidence:.2f}")

# Or load your custom model
custom_model = AutoModel.from_pretrained(
    "my-project-abc123/2",
    api_key="your_roboflow_api_key"
)
```

