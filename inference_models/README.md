# Inference Models

Welcome to **inference-models** - the next generation of computer vision inference engine from Roboflow.

## 🚀 What is `inference-models`?

`inference-models` is the library to make predictions from computer vision models provided by Roboflow — designed to 
be fast, reliable, and user-friendly. It offers:

- **Multi-Backend Support**: Run models with PyTorch, ONNX, TensorRT, or Hugging Face backends
- **Automatic Model Loading**: Smart model resolution and backend selection
- **Minimal Dependencies**: Composable extras system for installing only what you need
- **Unified Interface**: Consistent API across all model types and backends

> [!IMPORTANT]
> **Full Roboflow Platform Support**
> 
> **Run any model trained on [Roboflow](https://roboflow.com)** - your custom models work seamlessly alongside pretrained weights. For supported model architectures, Roboflow provides pretrained weights you can use without training your own model.

## 🛣️ Roadmap to Stable Release

We're actively working toward stabilizing `inference-models` and integrating it into the main `inference` package. The plan is to:

1. **Stabilize the API** - Finalize the core interfaces and ensure backward compatibility
2. **Integrate with `inference`** - Make `inference-models` available as a selectable backend in the `inference` package
3. **Production deployment** - Enable users to choose between the classic inference backend and the new `inference-models` backend
4. **Gradual migration** - Provide a smooth transition path for existing users

We're sharing this preview to gather valuable community feedback that will help us shape the final release. Your input is crucial in making this the best inference experience possible!

> [!NOTE]
> The `inference-models` package is approaching stability but is still in active development.
> * The core API is stabilizing, but minor changes may still occur
> * We're working toward backward compatibility guarantees
> * Production use is possible but we recommend thorough testing
> * For mission-critical systems, continue using the stable `inference` package until the official integration is complete

## ⚡ Quick Start

### Installation

> [!TIP]
> We recommend using `uv`
> 
> `uv` is a fast Python package installer and resolver. Install it with:
> 
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```
> 
> Learn more in the [uv documentation](https://docs.astral.sh/uv/).

**CPU installation:**
```bash
uv pip install inference-models
# or with pip
pip install inference-models
```

**GPU installation with ONNX and TensorRT support:**
```bash
uv pip install "inference-models[torch-cu128,onnx-cu12,trt10]" tensorrt
# or with pip
pip install "inference-models[torch-cu128,onnx-cu12,trt10]" tensorrt
```

> [!WARNING]
> **TensorRT Version Compatibility**
> 
> The `trt10` extra only works with TensorRT 10.x. We recommend installing the TensorRT version compatible with your target environment by specifying the exact version: `tensorrt==x.y.z`. For example, `tensorrt==10.12.0.36` for CUDA 12.x environments.

> [!NOTE]
> **Composable Dependencies**
> 
> The `inference-models` package uses a composable extras system - install only the backends and models you need. 
> See [Backends and Installation Options](https://roboflow.github.io/inference/inference_models/getting-started/backends/) for all available backends and their use cases.
> Learn more about this philosophy in [Principles and Assumptions](https://roboflow.github.io/inference/inference_models/getting-started/principles/).

### Usage

```python
import cv2
from inference_models import AutoModel
import supervision as sv

model = AutoModel.from_pretrained("rfdetr-base")

image = cv2.imread("<path-to-your-image>")
predictions = model(image)

# Visualize with supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image, predictions[0].to_supervision())
```

## 📚 Model selection optimized for your environment

When model is available in multiple backends, the same code works for all of them - `inference-models` automatically select best option based on your environment and installed dependencies. That strategy **maximizes performance** and **reduces your effort**.

**Available Backends:** PyTorch • ONNX • TensorRT • Hugging Face • MediaPipe

**Example: RFDetr Object Detection**

<details>
<summary><b>Auto-Selection</b></summary>

**Installation:**
```bash
# CPU
uv pip install "inference-models[onnx-cpu]"
# GPU
uv pip install "inference-models[torch-cu128,onnx-cu12,trt10]" tensorrt
```

**Code:**
```python
import cv2
from inference_models import AutoModel
import supervision as sv

# Automatically selects best available backend
model = AutoModel.from_pretrained("rfdetr-base")

image = cv2.imread("<path-to-your-image>")
predictions = model(image)

# Visualize with supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image, predictions[0].to_supervision())
```

</details>

<details>
<summary><b>PyTorch</b></summary>

**Installation:**
```bash
# CPU
uv pip install inference-models
# GPU
uv pip install "inference-models[torch-cu128]"
```

**Code:**
```python
import cv2
from inference_models import AutoModel
import supervision as sv

# Force PyTorch backend
model = AutoModel.from_pretrained("rfdetr-base", backend="torch")

image = cv2.imread("<path-to-your-image>")
predictions = model(image)

# Visualize with supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image, predictions[0].to_supervision())
```

</details>

<details>
<summary><b>ONNX</b></summary>

**Installation:**
```bash
# CPU
uv pip install "inference-models[onnx-cpu]"
# GPU
uv pip install "inference-models[onnx-cu12]"
```

**Code:**
```python
import cv2
from inference_models import AutoModel
import supervision as sv

# Force ONNX backend
model = AutoModel.from_pretrained("rfdetr-base", backend="onnx")

image = cv2.imread("<path-to-your-image>")
predictions = model(image)

# Visualize with supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image, predictions[0].to_supervision())
```

</details>

<details>
<summary><b>TensorRT</b></summary>

**Installation (Requires NVIDIA GPU):**
```bash
uv pip install "inference-models[trt10]" tensorrt
```

**Code:**
```python
import cv2
from inference_models import AutoModel
import supervision as sv

# Force TensorRT backend
model = AutoModel.from_pretrained("rfdetr-base", backend="trt")

image = cv2.imread("<path-to-your-image>")
predictions = model(image)

# Visualize with supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image, predictions[0].to_supervision())
```

</details>

> [!WARNING]
> **Backend Prediction Differences**
>
> Predictions may vary slightly between backends due to different numerical implementations. We aim for consistency, but minor differences are expected. Always validate performance for your specific use case.

## 🧠 Supported Model Architectures

- **RFDetr**
- **SAM models family**
- **Vision-Language Models** (Florence, PaliGemma, Qwen, SmolVLM, Moondream)
- **OCR** (DocTR, EasyOCR, TrOCR)
- **YOLO**
- and many more

For detailed model documentation, see [Supported Models](https://roboflow.github.io/inference/inference_models/models/).

## 🔧 Run your local models

Load your own model implementations from a local directory without contributing to the main library:

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "/path/to/my_custom_model",
    allow_local_code_packages=True
)
```

Your custom model directory structure:

```
my_custom_model/
├── model_config.json    # Model metadata
├── model.py            # Your model implementation
└── weights.pt          # Model weights (optional)
```

See [Load Models from Local Packages](https://roboflow.github.io/inference/inference_models/how-to/local-packages/) for complete details on creating custom model packages.

## 📖 Documentation

Visit the [full documentation](https://roboflow.github.io/inference/inference_models/) for:

- [Installation Guide](https://roboflow.github.io/inference/inference_models/getting-started/installation/)
- [Backends & Installation Options](https://roboflow.github.io/inference/inference_models/getting-started/backends/)
- [Quick Start](https://roboflow.github.io/inference/inference_models/getting-started/quickstart/)
- [Principles & Architecture](https://roboflow.github.io/inference/inference_models/getting-started/principles/)
- [Model Documentation](https://roboflow.github.io/inference/inference_models/models/)
- [API Reference](https://roboflow.github.io/inference/inference_models/api-reference/)

## 📄 License

The `inference-models` package is licensed under Apache 2.0. Individual models may have different licenses - see the [Supported Models](https://roboflow.github.io/inference/inference_models/models/) for details.

