---
hide:
  - navigation
  - toc
---

# 🚀 What is inference-models?

`inference-models` is the library to make predictions from computer vision models provided by Roboflow — designed to
be fast, reliable, and user-friendly. It offers:

- **Multi-Backend Support**: Run models with PyTorch, ONNX, TensorRT, or Hugging Face backends
- **Automatic Model Loading**: Smart model resolution and backend selection
- **Minimal Dependencies**: Composable extras system for installing only what you need
- **Behavior-Based Interfaces**: Models with similar behavior share consistent APIs; custom models can define their own
- **Full Roboflow Platform Support:** Run any model trained on [Roboflow](https://roboflow.com)

!!! info "Roadmap for `inference-models`"

    We are still making changes to the API and adding new features. API should be fairli stable already, but 
    it is advised to pin to specific version if you are using it in production and reveiw [our roadmap](./roadmap.md).


# 💻 Installation

=== "uv (recommended)"

    **CPU installation:**
    ```bash
    uv pip install inference-models
    ```

    `inference-models` can be installed with CUDA and TensorRT support - see [Installation Guide](getting-started/installation.md) for more options.

=== "pip"

    **CPU installation:**
    ```bash
    pip install inference-models
    ```

    `inference-models` can be installed with CUDA and TensorRT support - see [Installation Guide](getting-started/installation.md) for more options.

# 🏃‍➡️ Usage

=== "Pretrained Models"

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

=== "Your Roboflow Models"

    Load and run models trained on the [Roboflow platform](https://roboflow.com):

    ```python hl_lines="7-8"
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
- **Vision-Language Models** (Florence, PaliGemma, Qwen, SmolVLM, Moondream, [Gemma 4](models/gemma4.md))
- **OCR** (DocTR, EasyOCR, TrOCR)
- **YOLO**
- and many more

For detailed model documentation, see [Supported Models](models/index.md).

### Gemma 4 (Hugging Face multimodal)

[Gemma 4](models/gemma4.md) runs through the Hugging Face stack as `Gemma4HF` in `inference_models.models.gemma4.gemma4_hf` — multimodal prompts, optional Roboflow `inference_config.json`, and configurable visual token budgets. Defaults for generation are configurable via `INFERENCE_MODELS_GEMMA4_*` environment variables ([reference](how-to/environment-variables.md#gemma-4)).

# 🔧 Run your local models

Load your own model implementations from a local directory - models with architectures **not** in the main `inference-models` package. This is especially valuable for **production deployment** of custom models.
Find more information in [Load Models from Local Packages](how-to/local-packages.md).

```python
from inference_models import AutoModel

model = AutoModel.from_pretrained(
    "/path/to/my_custom_model",
    allow_local_code_packages=True
)
```

See [Load Models from Local Packages](how-to/local-packages.md) for complete details on creating custom model packages.

# 📄 License

The `inference-models` package is licensed under Apache 2.0. Individual models may have different licenses - see the [Supported Models](models/index.md) for details.

---

Ready to get started? Head to the [Quick Overview](getting-started/overview.md) →

