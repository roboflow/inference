# 🚀 What is inference-models?

`inference-models` is the library to make predictions from computer vision models provided by Roboflow — designed to
be fast, reliable, and user-friendly. It offers:

- **Multi-Backend Support**: Run models with PyTorch, ONNX, TensorRT, or Hugging Face backends
- **Automatic Model Loading**: Smart model resolution and backend selection
- **Minimal Dependencies**: Composable extras system for installing only what you need
- **Behavior-Based Interfaces**: Models with similar behavior share consistent APIs; custom models can define their own
- **Full Roboflow Platform Support:** Run any model trained on [Roboflow](https://roboflow.com)

> [!NOTE]
> **Roadmap for `inference-models`**
>
> We are still making changes to the API and adding new features. API should be fairly stable already, but
> it is advised to pin to specific version if you are using it in production and review [our roadmap](https://roboflow.github.io/inference/inference_models/roadmap/).

# 🛣️ Roadmap

We're actively working toward stabilizing `inference-models` and integrating it into the main `inference` package. The plan is to:

1. **Stabilize the API** - Finalize the core interfaces and ensure backward compatibility
2. **Integrate with `inference`** - Make `inference-models` available as a selectable backend in the `inference` package
3. **Production deployment** - Enable users to choose between the classic inference backend and the new `inference-models` backend
4. **Gradual migration** - Provide a smooth transition path for existing users

We're sharing this preview to gather valuable community feedback that will help us shape the final release. Your input is crucial in making this the best inference experience possible!

# 💻 Installation

**CPU installation:**
```bash
uv pip install inference-models
# or with pip
pip install inference-models
```

`inference-models` can be installed with CUDA and TensorRT support - see [Installation Guide](https://roboflow.github.io/inference/inference_models/getting-started/installation/) for more options.

# 🏃‍➡️ Usage

## Pretrained Models

Load and run a pretrained model:

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load pretrained model from Roboflow
model = AutoModel.from_pretrained("rfdetr-base")

# Run inference (works with numpy arrays or torch.Tensor)
image = cv2.imread("<path-to-your-image>")
predictions = model(image)

# Use with supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image, predictions[0].to_supervision())
```

## Your Roboflow Models

Load and run models trained on the [Roboflow platform](https://roboflow.com):

```python
import cv2
import supervision as sv
from inference_models import AutoModel

# Load your custom model from Roboflow
model = AutoModel.from_pretrained(
    "<your-project>/<version>",
    api_key="<your-api-key>"  # model access secured with API key
)

# Run inference (works with numpy arrays or torch.Tensor)
image = cv2.imread("<path-to-your-image>")
predictions = model(image)

# Use with supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image, predictions[0].to_supervision())
```



# 🧠 Supported Model Architectures

- **RFDetr**
- **SAM models family**
- **Vision-Language Models** (Florence, PaliGemma, Qwen, SmolVLM, Moondream)
- **OCR** (DocTR, EasyOCR, TrOCR)
- **YOLO**
- and many more

For detailed model documentation, see [Supported Models](https://roboflow.github.io/inference/inference_models/models/).

# 🔧 Run your local models

Load your own model implementations from a local directory - models with architectures **not** in the main `inference-models` package. This is especially valuable for **production deployment** of custom models.
Find more information in [Load Models from Local Packages](https://roboflow.github.io/inference/inference_models/how-to/local-packages/).

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "/path/to/my_custom_model",
    allow_local_code_packages=True
)
```

See [Load Models from Local Packages](https://roboflow.github.io/inference/inference_models/how-to/local-packages/) for complete details on creating custom model packages.

# 📄 License

The `inference-models` package is licensed under Apache 2.0. Individual models may have different licenses - see the [Supported Models](https://roboflow.github.io/inference/inference_models/models/) for details.

---

Ready to get started? Head to the [Quick Overview](https://roboflow.github.io/inference/inference_models/getting-started/overview/) →

