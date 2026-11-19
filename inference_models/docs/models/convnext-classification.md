# ConvNeXt - Classification

ConvNeXt is a convolutional architecture modernized with design choices from vision transformers: large depthwise kernels, LayerNorm and inverted bottlenecks. Roboflow trains it for single-label and multi-label classification from timm's pretrained weights.

## Overview

Key features:

- **Five sizes** - Nano, Tiny, Small, Base and Large, trained from timm's pretrained weights
- **Single-label and multi-label** classification
- **Image embeddings** - the exported head ends in a `head.fc` linear layer, so models serve the `image_embeddings` capability (`/infer/embeddings` and the Embedding Model block)
- **Same runtime as ResNet** - Roboflow packages ConvNeXt exactly like timm ResNet (ImageNet normalization, dynamic-size ONNX), so the ConvNeXt classes reuse the ResNet loading and inference code
- **Own default confidence** - multi-label models use `INFERENCE_MODELS_CONVNEXT_DEFAULT_CONFIDENCE`, which inherits from `INFERENCE_MODELS_DEFAULT_CONFIDENCE`, so a server can tune ConvNeXt and ResNet separately

## License

**Apache 2.0**

!!! info "Open Source License"
    ConvNeXt is licensed under Apache 2.0, making it free for both commercial and non-commercial use without restrictions.

    Learn more: [Apache 2.0 License](https://www.apache.org/licenses/LICENSE-2.0)

## Model IDs

There are no pre-trained ConvNeXt model IDs. Train a model on Roboflow and load it by its ID.

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
| **Serverless API (v2)** | ✅ [Deploy via hosted API](https://docs.roboflow.com/deploy/serverless-hosted-api-v2) |
| **Workflows** | ✅ Use in [Workflows](https://docs.roboflow.com/workflows) via the Classification and Embedding Model blocks |
| **Self-Hosting** | ✅ Deploy with `inference-models` |

## Usage Example

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "my-project-abc123/2",
    api_key="your_roboflow_api_key",
)
image = cv2.imread("path/to/image.jpg")

prediction = model(image)

top_class_id = prediction.class_id[0].item()
top_class = model.class_names[top_class_id]
confidence = prediction.confidence[0][top_class_id].item()

print(f"Class: {top_class}")
print(f"Confidence: {confidence:.2f}")
```
