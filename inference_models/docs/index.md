---
hide:
  - navigation
---

# Inference Models

Welcome to the **inference-models** documentation - the next generation of computer vision inference from Roboflow.

!!! warning "Experimental Preview"
    The `inference-models` package is an **experimental preview** of upcoming inference capabilities.
    
    * Features may change, break, or be removed without notice.
    * We **do not guarantee backward compatibility** between releases.
    
    ❗ **We strongly advise against** using `inference-models` in production systems - for such purposes 
    please **continue to use the stable `inference` package.**

## 🚀 What is inference-models?

At Roboflow, we're building a new generation of `inference` — designed to be faster, more reliable, and more user-friendly. The `inference-models` library represents this evolution with:

- **Multi-Backend Support**: Run models with PyTorch, ONNX, TensorRT, or Hugging Face backends
- **Automatic Model Loading**: Smart model resolution and backend selection
- **Minimal Dependencies**: Composable extras system for installing only what you need
- **Unified Interface**: Consistent API across all model types and backends
- **Production-Ready**: Optimized for both development and deployment

## ⚡ Quick Start

Install the package:

```bash
# CPU installation
pip install inference-models

# GPU installation with ONNX and TensorRT support
pip install "inference-models[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
```

Load and run a model:

```python
from inference_models import AutoModel
import cv2

# Load model from Roboflow
model = AutoModel.from_pretrained("rfdetr-base")

# Run inference
image = cv2.imread("path/to/image.jpg")
predictions = model(image)

# Use with supervision
import supervision as sv
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image.copy(), predictions[0].to_supervision())
```

## 📚 Key Features

### Multi-Backend Architecture

Run the same model with different backends depending on your environment:

- **PyTorch**: Default backend, maximum flexibility
- **ONNX**: Cross-platform compatibility, good performance
- **TensorRT**: Maximum GPU performance
- **Hugging Face**: Access to transformer-based models

The library automatically selects the best available backend based on your installation and hardware.

### Smart Model Loading

The `AutoModel` class handles all the complexity:

- Automatic backend selection
- Model package resolution
- Dependency management
- Caching and optimization

### Composable Dependencies

Install only what you need with extras:

```bash
# ONNX backend for CPU
pip install "inference-models[onnx-cpu]"

# Full GPU setup with TensorRT
pip install "inference-models[torch-cu128,onnx-cu12,trt10]"

# Additional models
pip install "inference-models[mediapipe,grounding-dino]"
```

## 🧠 Supported Models

The library supports a wide range of computer vision tasks:

- **Object Detection**: YOLOv8-12, RFDetr, YOLO-NAS, Grounding DINO, OWLv2, YOLO-World
- **Instance Segmentation**: YOLOv8/11 Seg, YOLACT
- **Classification**: ResNet, ViT, DINOv3
- **Embeddings**: CLIP, Perception Encoder
- **OCR**: DocTR, EasyOCR, TrOCR
- **Segmentation**: SAM, SAM2, SAM2 Real-Time
- **Vision-Language**: Florence-2, PaliGemma, Qwen2.5-VL, Qwen3-VL, SmolVLM, Moondream2
- **Depth Estimation**: Depth Anything V2/V3
- **Specialized**: L2CS (Gaze), MediaPipe Face Detection

See the [Models Overview](models/index.md) for complete details.

## 📖 Documentation Structure

- **[Getting Started](getting-started/installation.md)**: Installation, quick start, and core principles
- **[Auto-Loading](auto-loading/overview.md)**: Understanding the automatic model loading system
- **[Models](models/index.md)**: Detailed documentation for each supported model
- **[How-To Guides](how-to/local-packages.md)**: Practical guides for common tasks
- **[Contributors](contributors/architecture.md)**: Architecture and development guides
- **[API Reference](api-reference/)**: Complete API documentation

## 🤝 Community & Support

This is an early-stage project, and we're sharing initial versions to gather valuable community feedback. Your input will help us shape and steer this initiative in the right direction.

- **GitHub Issues**: [Report bugs or request features](https://github.com/roboflow/inference/issues/)
- **Discussions**: [Ask questions and share ideas](https://github.com/roboflow/inference/discussions)
- **Discord**: [Join our community](https://discord.gg/roboflow)

## 🔗 Related Projects

- **[inference](https://github.com/roboflow/inference)**: The stable, production-ready inference library
- **[supervision](https://github.com/roboflow/supervision)**: Computer vision utilities for working with predictions
- **[Roboflow](https://roboflow.com)**: Train and deploy custom computer vision models

## 📄 License

The `inference-models` package is licensed under Apache 2.0. Individual models may have different licenses - see the [Models Overview](models/index.md) for details.

---

Ready to get started? Head to the [Installation Guide](getting-started/installation.md) →

