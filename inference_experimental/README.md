# Experimental version of inference

## 🚀 Introducing `inference-exp` - the evolution of `inference`

At Roboflow, we’re taking a bold step toward a new generation of `inference` — designed to be faster, 
more reliable, and more user-friendly. With this vision in mind, we’re building a new library called `inference-exp`.

This is an early-stage project, and we’re sharing initial versions to gather valuable community feedback. 
Your input will help us shape and steer this initiative in the right direction.

We’re excited to have you join us on this journey — let’s build something great together! 🤝

> [!CAUTION]
> The `inference-exp` package **is an experimental preview** of upcoming inference capabilities.
> **🔧 What this means:**
> * Features may change, break, or be removed without notice.
> * We **do not guarantee backward compatibility** between releases.
> * We are publishing this to PyPI only **for preview and feedback purposes.**
> * Although `inference-exp` is located in the `inference` codebase, it is not included in any production build and
> its lifecycle is completely independent of the official `inference` package releases.
> 
> ❗ **We strongly advise against** using `inference-exp` in production systems or building integrations on top of it.
> For production use and official model deployment, please **continue to use the stable `inference` package.**

## 📜 Principles and Assumptions

* We define a **model** as weights trained on a dataset, which can be exported or compiled into multiple equivalent 
**model packages**, each optimized for specific environments (e.g., speed, flexibility).

* The new inference library is **multi-backend**, able to run model packages in different formats 
depending on the installed dependencies - with the scope of supported models dependent on the choice of package 
*extras* made during installation

* We aim to keep the **extra dependencies minimal** while covering as broad a range of models as possible.

* By default, we include **PyTorch** and **Hugging Face Transformers**; optional extras are available for 
**TensorRT (TRT)** and **ONNX** backends, with a runtime preference order: TRT → Torch → ONNX. We wish new models
are mostly based on Torch.

* Backend selection happens **dynamically at runtime**, based on model metadata and environment checks, 
but can be fully overridden by the user when needed.

## ⚡ Installation

> [!TIP]
> We recommend using `uv` to install `inference-exp`. To install the tool, follow 
> [official guide](https://docs.astral.sh/uv/getting-started/installation/) or use the snippet below:
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```


To install `inference-exp` **with TRT and ONNX** on GPU server with base CUDA libraries available run the following 
command:

```bash
uv pip install "inference-exp[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
```
> [!TIP]
> To avoid clashes with external packages, `pyproject.toml` defines quite loose restrictions for the dependent packages.
> Some packages, like `tensorrt` are good to be kept under more strict control (as some TRT engines will only work 
> when there is an exact match of environment that runs the model with the one that compiled it) - that's why we 
> recommend fixing `tensorrt` version to the one we currently use to compile TRT artefacts.
> 
> Additionally, library defines set of `torch-*` extras which, thanks to `uv` deliver extra packages indexes adjusted 
> for specific CUDA version: `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126`.

For CPU installations, we recommend the following commands:
```bash
# to install with ONNX backend
uv pip install "inference-exp[onnx-cpu]"
# or - to install only base dependencies
uv pip install inference-exp
```

> [!NOTE] 
> Using `uv pip install ...` or `pip install`, it is possible to get non-reproducible builds (as `pyproject.toml` 
> defines quite loose restrictions for the dependent packages). If you care about strict control of dependencies - 
> follow the installation method based on `uv.lock` which is demonstrated in official [docker builds](./dockerfiles) 
> of the library.

## 📖 Basic Usage
```python
from inference_exp import AutoModel
import cv2
import supervision as sv

# loads model from Roboflow API (loading from local dir also available)
model = AutoModel.from_pretrained("yolov8n-640")  
image = cv2.imread("<path-to-your-image>")
predictions = model(image)[0]

# integration with supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image.copy(), predictions.to_supervision())
```

## 🔌 Extra Dependencies

### Backends
| Backend | Extras                                                                        | Description                                                                                                                                                                                                                                                                                                                   |
|---------|-------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| PyTorch | `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` | Provide specific variants of `torch` to match installed CUDA version, only works with `uv` which is capable of reading extra indexes from `pyproject.toml`, when using with `pip`, use `--extra-index-url`. By default, CPU version of `torch` is installed with the library. Torch backend is a default one for the library. |
| ONNX    | `onnx-cpu`, `onnx-cu118`, `onnx-cu12`, `onnx-jp6-cu126`                       | Provide specific variants of `onnxruntime`. only works with `uv` which is capable of reading extra indexes from `pyproject.toml`, when using with `pip`, use `--extra-index-url`. This extras is not installed by default and is not required, but enables wide variety of models trained on Roboflow Platform.               |
| TRT     | `trt10`                                                                       | Provide specific variants of `tensorrt`, only works on GPU servers. Jetson installations should fall back to pre-compiled package shipped with Jetpack.                                                                                                                                                                       |


### Additional models / capabilities
| Extras           | Description                                                                                        |
|------------------|----------------------------------------------------------------------------------------------------|
| `mediapipe`      | Enables MediaPipe models, including Face Detector                                                  |
| `grounding-dino` | Enables Grounding Dino model                                                                       |
| `flash-attn`     | *EXPERIMENTAL:* Installs `flash-attn` for faster LLMs/VLMs - usually requires extensive compilation |
| `test`           | Test dependencies                                                                                  |


## 🧠 Models

> [!IMPORTANT] 
> If you see a bug in model implementation or loading mechanism - create 
> [new issue](https://github.com/roboflow/inference/issues/) tagging it with `inference-exp-bug`.
> 
> Additionally, We are working hard to extend pool of supported models - suggestions on new models to be added 
> appreciated 🤝


Below there is a table showcasing models that are supported, with the hints regarding extra dependencies that 
are required.

| Architecture       | Task Type               | Supported variants |
|--------------------|-------------------------|--------------------|
| RFDetr             | `object-detection`      | TRT, Torch         |
| YOLO v8            | `object-detection`      | ONNX, TRT          |
| YOLO v8            | `instance-segmentation` | ONNX, TRT          |
| YOLO v9            | `object-detection`      | ONNX, TRT          |
| YOLO v10           | `object-detection`      | ONNX, TRT          |
| YOLO v11           | `object-detection`      | ONNX, TRT          |
| YOLO v11           | `instance-segmentation` | ONNX, TRT          |
| Perception Encoder | `embedding`             | Torch              |
