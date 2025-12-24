# Experimental version of inference

## 🚀 Introducing `inference-models` - the evolution of `inference`

At Roboflow, we’re taking a bold step toward a new generation of `inference` — designed to be faster, 
more reliable, and more user-friendly. With this vision in mind, we’re building a new library called `inference-models`.

This is an early-stage project, and we’re sharing initial versions to gather valuable community feedback. 
Your input will help us shape and steer this initiative in the right direction.

We’re excited to have you join us on this journey — let’s build something great together! 🤝

> [!CAUTION]
> The `inference-models` package **is an experimental preview** of upcoming inference capabilities.
> * Features may change, break, or be removed without notice.
> * We **do not guarantee backward compatibility** between releases.
> 
> ❗ **We strongly advise against** using `inference-models` in production systems - for such purposes 
> please **continue to use the stable `inference` package.**

## ⚡ Installation

> [!TIP]
> We recommend using `uv` to install `inference-models`. To install the tool, follow 
> [official guide](https://docs.astral.sh/uv/getting-started/installation/) or use the snippet below:
> ```bash
> curl -LsSf https://astral.sh/uv/install.sh | sh
> ```

Use the following command to install `inference-models` on **CPU machine 💻** (below you can find more advanced options):

```bash
uv pip install inference-models
# or - if you use pip
pip install inference-models
```

<details>
<summary>👉 GPU installation</summary>

As you may learn from [📜 Principles and Assumptions](#-principles-and-assumptions), `inference-models` is designed to 
compose the build out of different [extras](#-extra-dependencies) defined for the package. Some extras bring new 
models, while others - ability to run models created for specific backend. To get the most out of the installation
on GPU machine, we recommend including TRT and ONNX extensions, as well as select `torch-cu*` extras to install 
torch compliant with version of CUDA installed on the machine. ONNX backend is particularly important when running
models trained on Roboflow platform.

```bash
uv pip install "inference-models[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
# or - if you use pip
pip install "inference-models[torch-cu128,onnx-cu12,trt10]" "tensorrt==10.12.0.36"
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
uv pip install "inference-models[onnx-cpu]"
# or - to install only base dependencies
uv pip install inference-models
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
from inference_models import AutoModel
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
uv pip install "inference-models[extras-name-1,extras-name-1]"
# or - if you use pip
pip install "inference-models[extras-name-1,extras-name-2]"
```

In case of `inference-models`, extras bring either additional **backends** (dependencies to run AI models of different type, 
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

#### Special Installation: SAM2 Real-Time

`sam2 real time` requires a Git-based dependency that cannot be distributed via PyPI. To use SAM2 real-time capabilities, you need to manually install it after installing `inference-models`:

```bash
# First, install inference-models with your desired extras (e.g., torch-cu124)
pip install "inference-models[torch-cu124]"

# Then, install SAM2 real-time from GitHub
pip install git+https://github.com/Gy920/segment-anything-2-real-time.git
```

For development environments:
```bash
# First sync the project
uv sync --dev

# Then manually install SAM 2 from the GitHub repository
# Note: The package installs as "SAM 2" (with a space)
uv pip install git+https://github.com/Gy920/segment-anything-2-real-time.git
```

> [!NOTE]
> Due to PyPI restrictions on Git dependencies, the SAM2 real-time package must be installed separately from the GitHub repository. The package will be installed with the name "SAM 2" (with a space).

> [!IMPORTANT]  
> Not all extras are possible to be installed together in a single environment. We try to make the extras as composable 
> as possible, but **this will not always be possible**, and sometimes you need to choose which extras are to be 
> installed.


## 🧠 Models

> [!IMPORTANT] 
> If you see a bug in model implementation or loading mechanism - create 
> [new issue](https://github.com/roboflow/inference/issues/) tagging it with `inference-models-bug`.
> 
> Additionally, We are working hard to extend pool of supported models - suggestions on new models to be added 
> appreciated 🤝


Below there is a table showcasing models that are supported, with the hints regarding extra dependencies that 
are required.

| Architecture       | Task Type               | Supported backends |
|--------------------|-------------------------|--------------------|
| RFDetr             | `object-detection`      | `trt`, `torch`     |
| YOLO v8            | `object-detection`      | `onnx`, `trt`      |
| YOLO v8            | `instance-segmentation` | `onnx`, `trt`      |
| YOLO v9            | `object-detection`      | `onnx`, `trt`      |
| YOLO v10           | `object-detection`      | `onnx`, `trt`      |
| YOLO v11           | `object-detection`      | `onnx`, `trt`      |
| YOLO v11           | `instance-segmentation` | `onnx`, `trt`      |
| Perception Encoder | `embedding`             | `torch`            |
| CLIP               | `embedding`             | `torch`, `onnx`     |


### Registered pre-trained weights

Below you can find a list of model IDs registered in Roboflow weights provider (along with notes about access rights).

* `public-open` - available without Roboflow API key, but under licenses for specific model 

* `public-api-key-gated` - available for everyone with Roboflow API key

**Models:**

<details>
<summary>👉 <b>RFDetr</b></summary>

**Access level:** `public-open`

**License:**  [Apache 2.0](inference_models/models/rfdetr/LICENSE.txt)

The following model IDs are registered:

* `rfdetr-base` (trained on COCO dataset)

* `rfdetr-base` (trained on COCO dataset)

</details>

<details>
<summary>👉 <b>YOLO v8</b></summary>

**Access level:** `public-open`

**License:**  [AGPL](inference_models/models/yolov8/LICENSE.txt)

The following model IDs are registered for **object detection** task:

* `yolov8n-640` (trained on COCO dataset)

* `yolov8n-1280` (trained on COCO dataset)

* `yolov8s-640` (trained on COCO dataset)

* `yolov8s-1280` (trained on COCO dataset)

* `yolov8m-640` (trained on COCO dataset)

* `yolov8m-1280` (trained on COCO dataset)

* `yolov8l-640` (trained on COCO dataset)

* `yolov8l-1280` (trained on COCO dataset)

* `yolov8x-640` (trained on COCO dataset)

* `yolov8x-1280` (trained on COCO dataset)


The following model IDs are registered for **instance segmentation** task:

* `yolov8n-seg-640` (trained on COCO dataset)

* `yolov8n-seg-1280` (trained on COCO dataset)

* `yolov8s-seg-640` (trained on COCO dataset)

* `yolov8s-seg-1280` (trained on COCO dataset)

* `yolov8m-seg-640` (trained on COCO dataset)

* `yolov8m-seg-1280` (trained on COCO dataset)

* `yolov8l-seg-640` (trained on COCO dataset)

* `yolov8l-seg-1280` (trained on COCO dataset)

* `yolov8x-seg-640` (trained on COCO dataset)

* `yolov8x-seg-1280` (trained on COCO dataset)

</details>


<details>
<summary>👉 <b>YOLO v10</b></summary>

**Access level:** `public-open`

**License:**  [AGPL](inference_models/models/yolov10/LICENSE.txt)

The following model IDs are registered for **object detection** task:

* `yolov10n-640` (trained on COCO dataset)

* `yolov10s-640` (trained on COCO dataset)

* `yolov10m-640` (trained on COCO dataset)

* `yolov10b-640` (trained on COCO dataset)

* `yolov10l-640` (trained on COCO dataset)

* `yolov10x-640` (trained on COCO dataset)

</details>


<details>
<summary>👉 <b>Perception Encoder</b></summary>

**Access level:** `public-open`

**License:**  [FAIR Noncommercial Research License](inference_models/models/perception_encoder/vision_encoder/LICENSE.weigths.txt)

The following model IDs are registered:

* `perception-encoder/PE-Core-B16-224`

* `perception-encoder/PE-Core-G14-448`

* `perception-encoder/PE-Core-L14-336`

</details>

<details>
<summary>👉 <b>CLIP</b></summary>

**Access level:** `public-open`

**License:**  [MIT](inference_models/models/clip/LICENSE.txt)

The following model IDs are registered:

* `clip/RN50`

* `clip/RN101`

* `clip/RN50x16`
 
* `clip/RN50x4`
 
* `clip/RN50x64`
 
* `clip/ViT-B-16`
 
* `clip/ViT-B-32`
 
* `clip/ViT-L-14-336px`
 
* `clip/ViT-L-14`

</details>

## 📜 Citations

```
@article{bolya2025PerceptionEncoder,
  title={Perception Encoder: The best visual embeddings are not at the output of the network},
  author={Daniel Bolya and Po-Yao Huang and Peize Sun and Jang Hyun Cho and Andrea Madotto and Chen Wei and Tengyu Ma and Jiale Zhi and Jathushan Rajasegaran and Hanoona Rasheed and Junke Wang and Marco Monteiro and Hu Xu and Shiyu Dong and Nikhila Ravi and Daniel Li and Piotr Doll{\'a}r and Christoph Feichtenhofer},
  journal={arXiv:2504.13181},
  year={2025}
}
```
