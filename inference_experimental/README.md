# Experimental version of inference

## 🚀 Introducing `inference-exp` - the evolution of `inference`

At Roboflow, we’re taking a bold step toward a new generation of `inference` — designed to be faster, 
more reliable, and more user-friendly. With this vision in mind, we’re building a new library called `inference-exp`.

This is an early-stage project, and we’re sharing initial versions to gather valuable community feedback. 
Your input will help us shape and steer this initiative in the right direction.

We’re excited to have you join us on this journey — let’s build something great together! 🤝

> [!CAUTION]
> The `inference-exp` package **is an experimental preview** of upcoming inference capabilities.
> * Features may change, break, or be removed without notice.
> * We **do not guarantee backward compatibility** between releases.
> 
> ❗ **We strongly advise against** using `inference-exp` in production systems - for such purposes 
> please **continue to use the stable `inference` package.**

## ⚡ Installation

> [!TIP]
> We recommend using `uv` to install `inference-exp`. To install the tool, follow 
> [official guide](https://docs.astral.sh/uv/getting-started/installation/) or use the snippet below:
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

Use the following command to install `inference-exp` on **CPU machine 💻** (below you can find more advanced options):

```bash
uv pip install inference-exp
# or - if you use pip
pip install inference-exp
```

<details>
<summary>👉 GPU installation</summary>

As you may learn from [📜 Principles and Assumptions](#-principles-and-assumptions), `inference-exp` is designed to 
compose the build out of different [extras](#-extra-dependencies) defined for the package. Some extras bring new 
models, while others - ability to run models created for specific backend. To get the most out of the installation
on GPU machine, we recommend including TRT and ONNX extensions, as well as select `torch-cu*` extras to install 
torch compliant with version of CUDA installed on the machine. ONNX backend is particularly important when running
models trained on Roboflow platform.

```bash
uv pip install "inference-exp[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
# or - if you use pip
pip install "inference-exp[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
```

> To avoid clashes with external packages, `pyproject.toml` defines quite loose restrictions for the dependent packages.
> Some packages, like `tensorrt` are good to be kept under more strict control (as some TRT engines will only work 
> when there is an exact match of environment that runs the model with the one that compiled it) - that's why we 
> recommend fixing `tensorrt` version to the one we currently use to compile TRT artefacts.
> 
> Additionally, library defines set of `torch-*` extras which, thanks to `uv` deliver extra packages indexes adjusted 
> for specific CUDA version: `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126`.

</details>

<details>
<summary>👉 CPU installation - enabling <b>models trained with Roboflow</b></summary>

For CPU installations, we recommend installing ONNX backed, as the majority of models trained on Roboflow platform 
are exported to ONNX and not available:
```bash
# to install with ONNX backend
uv pip install "inference-exp[onnx-cpu]"
# or - to install only base dependencies
uv pip install inference-exp
```

</details>


<details>
<summary>👉 Reproducibility of installation</summary>

> Using `uv pip install ...` or `pip install`, it is possible to get non-reproducible builds (as `pyproject.toml` 
> defines quite loose restrictions for the dependent packages). If you care about strict control of dependencies - 
> follow the installation method based on `uv.lock` which is demonstrated in official [docker builds](./dockerfiles) 
> of the library.

</details>


## 📖 Basic Usage
```python
from inference_exp import AutoModel
import cv2
import supervision as sv

# loads model from Roboflow API (loading from local dir also available)
model = AutoModel.from_pretrained("rfdetr-base")  
image = cv2.imread("<path-to-your-image>")
predictions = model(image)

# integration with supervision
annotator = sv.BoxAnnotator()
annotated = annotator.annotate(image.copy(), predictions[0].to_supervision())
```

> [!TIP]
> Model **failed to load,** and you see error prompting you to **install additional dependencies**?
> 
> Take a look at [📜 Principles and Assumptions](#-principles-and-assumptions) to understand why this happens and 
> navigate to [extras](#-extra-dependencies) section to find out which extra dependency you need to install. 
> The common issue is lack of ONNX backend required to run models trained on Roboflow platform.


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


## 🔌 Extra Dependencies

Extras dependencies are optional features of the package that can be installed with:

```bash
uv pip install "inference-exp[extras-name-1,extras-name-1]"
# or - if you use pip
pip install "inference-exp[extras-name-1,extras-name-2]"
```

In case of `inference-exp`, extras bring either additional **backends** (dependencies to run AI models of different type, 
like TensorRT engines) or additional **models**. 

### Backends
| Extras names                                                                  | Backend | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
|-------------------------------------------------------------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `torch-cu118`, `torch-cu124`, `torch-cu126`, `torch-cu128`, `torch-jp6-cu126` | PyTorch | Provide specific variants of `torch` to match installed CUDA version, only works with `uv` which is capable of reading extra indexes from `pyproject.toml`, when using with `pip`, use `--extra-index-url`. By default, CPU version of `torch` is installed with the library. Torch backend is a default one for the library. Extras named `torch-cu*` are relevant for GPU servers with certain CUDA version, whereas extras like `torch-jp6-cu126` are to be installed on Jetson with specific Jetpack and CUDA versions. |
| `onnx-cpu`, `onnx-cu118`, `onnx-cu12`, `onnx-jp6-cu126`                       | ONNX    | Provide specific variants of `onnxruntime`. only works with `uv` which is capable of reading extra indexes from `pyproject.toml`, when using with `pip`, use `--extra-index-url`. This extras is not installed by default and is not required, but enables wide variety of models trained on Roboflow Platform. Extras named `onnx-cu*` are relevant for GPU servers with certain CUDA version, whereas extras like `onnx-jp6-cu126` are to be installed on Jetson with specific Jetpack and CUDA versions.                 |
| `trt10`                                                                       | TRT     | Provide specific variants of `tensorrt`, only works on GPU servers. Jetson installations should fall back to pre-compiled package shipped with Jetpack.                                                                                                                                                                                                                                                                                                                                                                     |


### Additional models / capabilities
| Extras           | Description                                                                                        |
|------------------|----------------------------------------------------------------------------------------------------|
| `mediapipe`      | Enables MediaPipe models, including Face Detector                                                  |
| `grounding-dino` | Enables Grounding Dino model                                                                       |
| `flash-attn`     | *EXPERIMENTAL:* Installs `flash-attn` for faster LLMs/VLMs - usually requires extensive compilation |
| `test`           | Test dependencies                                                                                  |

> [!IMPORTANT]  
> Not all extras are possible to be installed together in a single environment. We try to make the extras as composable 
> as possible, but **this will not always be possible**, and sometimes you need to choose which extras are to be 
> installed.


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
| CLIP               | `embedding`             | Torch, ONNX        |


### Registered pre-trained weights

Below you can find a list of model IDs registered in Roboflow weights provider (along with notes about access rights).

* `public-open` - available without Roboflow API key, but under licenses for specific model 

* `public-api-key-gated` - available for everyone with Roboflow API key

**Models:**

* **RFDetr:** `rfdetr-base` (COCO), `rfdetr-large` (COCO) - all `public-open` - [license](./inference_exp/models/rfdetr/LICENSE.txt)
* **YOLO v8 (object-detection):** `yolov8n-640` (COCO), `yolov8n-1280` (COCO), `yolov8s-640` (COCO), `yolov8s-1280` (COCO), `yolov8m-640` (COCO), `yolov8m-1280` (COCO), `yolov8l-640` (COCO), `yolov8l-1280` (COCO), `yolov8x-640` (COCO), `yolov8x-1280` (COCO) - all `public-open` - [license](./inference_exp/models/yolov8/LICENSE.txt)
* **YOLO v8 (instance-segmentation):** `yolov8n-seg-640` (COCO), `yolov8n-seg-1280` (COCO), `yolov8s-seg-640` (COCO), `yolov8s-seg-1280` (COCO), `yolov8m-seg-640` (COCO), `yolov8m-seg-1280` (COCO), `yolov8l-seg-640` (COCO), `yolov8l-seg-1280` (COCO), `yolov8x-seg-640` (COCO), `yolov8x-seg-1280` (COCO) - all `public-open` - [license](./inference_exp/models/yolov8/LICENSE.txt)
* **YOLO v10 (object-detection):** `yolov10n-640` (COCO), `yolov10s-640` (COCO), `yolov10m-640` (COCO), `yolov10b-640` (COCO), `yolov10l-640` (COCO), `yolov10x-640` (COCO)  - all `public-open` - [license](./inference_exp/models/yolov10/LICENSE.txt)
* **Perception Encoder:** `perception-encoder/PE-Core-B16-224`, `perception-encoder/PE-Core-G14-448`, `perception-encoder/PE-Core-L14-336` - all `public-open` - [license](./inference_exp/models/perception_encoder/vision_encoder/LICENSE.weigths.txt)
* **CLIP:** `clip/RN50`, `clip/RN101`, `clip/RN50x16`, `clip/RN50x4`, `clip/RN50x64`, `clip/ViT-B-16`, `clip/ViT-B-32`, `clip/ViT-L-14-336px`, `clip/ViT-L-14` - all `public-open` - [license](./inference_exp/models/clip/LICENSE.txt)

## 📜 Citations

```
@article{bolya2025PerceptionEncoder,
  title={Perception Encoder: The best visual embeddings are not at the output of the network},
  author={Daniel Bolya and Po-Yao Huang and Peize Sun and Jang Hyun Cho and Andrea Madotto and Chen Wei and Tengyu Ma and Jiale Zhi and Jathushan Rajasegaran and Hanoona Rasheed and Junke Wang and Marco Monteiro and Hu Xu and Shiyu Dong and Nikhila Ravi and Daniel Li and Piotr Doll{\'a}r and Christoph Feichtenhofer},
  journal={arXiv:2504.13181},
  year={2025}
}
```
