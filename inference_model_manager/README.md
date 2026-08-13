# inference-model-manager

Manages model lifecycle (load, unload, evict) and dispatches inference requests. Sits between your code and `inference-models` — you don't call models directly.

## Install

Requires Python 3.10–3.12. From this directory:

```bash
python -m venv .venv
source .venv/bin/activate
pip install uv

# For development: install inference-models editable first
# uv pip install -e "../inference_models"

# CPU (torch + ONNX)
uv pip install -e ".[torch-cpu,onnx-cpu]"

# CUDA 12.4
uv pip install -e ".[torch-cu124,onnx-cu12]"

# CUDA 12.6 + TRT (Jetson JP6)
uv pip install -e ".[torch-jp6-cu126,onnx-jp6-cu126]"
```

Extras cascade to `inference-models`.

## Backends

Two backends:

- **Direct** — model runs in your process. Simple, good for scripts and notebooks.
- **Subprocess** — model runs in a separate process with shared memory transport. Process isolation, GPU fault containment, zero-copy I/O.

## Direct backend

Model loads and runs in the same process. Fastest for single-model use.

```python
import urllib.request
import imagecodecs
from inference_model_manager.model_manager import ModelManager

mm = ModelManager()
mm.load("yolov8n-640", api_key="YOUR_KEY", backend="direct")

# imagecodecs returns RGB. Pass input_color_format="rgb" so model
# pre-processing knows not to flip channels (default assumes BGR).
image_bytes = urllib.request.urlopen("https://media.roboflow.com/dog.jpeg").read()
image = imagecodecs.imread(image_bytes)
result = mm.process("yolov8n-640", images=image, input_color_format="rgb")
print(result)
# {"type": "roboflow-object-detection-compact-v1", "class_names": [...], "xyxy": ..., ...}

mm.shutdown()
```

## Subprocess backend

Model loads in a child process. Communicates via shared memory pool — images written to SHM slots, results read back. No serialization on the hot path for image data.

```python
import urllib.request
import imagecodecs
from inference_model_manager.model_manager import ModelManager

mm = ModelManager()
mm.load("yolov8n-640", api_key="YOUR_KEY", backend="subprocess")

# Same API — backend difference is transparent.
image_bytes = urllib.request.urlopen("https://media.roboflow.com/dog.jpeg").read()
image = imagecodecs.imread(image_bytes)
result = mm.process("yolov8n-640", images=image, input_color_format="rgb")
print(result)
# Same typed dict output as direct backend.

mm.shutdown()  # kills worker process, frees SHM
```

Use subprocess when you need:
- Process isolation (model crash doesn't kill your app)
- Multiple models on one GPU without GIL contention
- Worker-side batching (accumulate requests, decode + infer in one GPU call)

## Color format

Models default to BGR input (OpenCV convention). Pass `input_color_format` if your source is different:

| Source | Format | Pass to `process()` |
|--------|--------|---------------------|
| `imagecodecs.imread()` | RGB | `input_color_format="rgb"` |
| `cv2.imread()` | BGR | nothing (default) |
| `PIL.Image` → `np.array()` | RGB | `input_color_format="rgb"` |
| torch tensor (CHW) | RGB | nothing (tensor default is RGB) |

## Registering a model in the registry

Models in `inference-models` work standalone — no changes needed there. To make a model available through model manager (task dispatch, validation, typed serialization), add an entry to `registry_defaults.py`.

### Case 1: Model inherits from a registered base class

If your model inherits from `ObjectDetectionModel`, `ClassificationModel`, `InstanceSegmentationModel`, etc. — **nothing to do**. The registry matches by class name via MRO. Your model inherits the base class entry automatically.

```python
# inference_models/models/my_detector/my_detector.py
class MyDetector(ObjectDetectionModel):
    def infer(self, images, **kwargs):
        ...
```

This works out of the box with `mm.process("my-detector", images=img)`.

### Case 2: New base class or model with unique tasks

Add entries to `_TASK_CONFIGS` in `registry_defaults.py`. Each entry is a tuple:

```
(task_name, method_name, is_default, params_dict, validator_name, serializer_name, response_type)
```

Example — a model with two tasks:

```python
# In registry_defaults.py _TASK_CONFIGS dict:
"MyCustomModel": [
    ("generate", "generate_output", True,
     {
         "images": {"type": "image", "required": True},
         "prompt": {"type": "str", "required": True},
         "temperature": {"type": "float", "required": False, "default": 0.7},
     },
     "validate_images_and_prompt", "serialize_text",
     "roboflow-text-v1"),
    ("embed", "embed_images", False,
     {"images": {"type": "image", "required": True}},
     "validate_images_required", "serialize_embeddings",
     "roboflow-embeddings-compact-v1"),
],
```

Reusable param fragments (`_P_IMAGES`, `_P_IMAGES_PROMPT`, `_K_OD`, etc.) are defined at the top of `registry_defaults.py`. Use `_p()` to merge them:

```python
"MyDetector": [
    ("infer", "infer", True, _p(_P_IMAGES, _K_OD),
     "validate_images_required", "serialize_detections_compact",
     "roboflow-object-detection-compact-v1"),
],
```

Fields:
- **task_name** — what users pass as `task=` param (e.g. `mm.process("model", task="embed")`)
- **method_name** — actual method on the model class to call (can differ from task_name)
- **is_default** — exactly one task must be `True`; used when `task=None`
- **params_dict** — `{name: {type, required, default?}}` — exposed in stats/interface for API discovery
- **validator_name** — function from `validators.py` (e.g. `"validate_images_required"`)
- **serializer_name** — function from `serializers_typed.py` (e.g. `"serialize_text"`)
- **response_type** — type string for JSON response envelope

If your model inherits from a registered base class but has different params (e.g. different defaults), add a concrete class entry — MRO picks it up first.

### Case 3: Custom validator or serializer

Add to `validators.py` or `serializers_typed.py`:

```python
# validators.py
def validate_my_custom_input(kwargs: dict) -> dict:
    if "images" not in kwargs:
        raise ValueError("'images' required")
    if "language" not in kwargs:
        raise ValueError("'language' required for this model")
    return kwargs
```

```python
# serializers_typed.py
def serialize_my_custom_output(output, model) -> dict:
    return {
        "type": "my-custom-output-v1",
        "result": output.result,
        "metadata": output.metadata,
    }
```

Then reference by name in `_TASK_CONFIGS`:

```python
"MyCustomModel": [
    ("infer", "infer", True, ["images", "language"],
     "validate_my_custom_input", "serialize_my_custom_output",
     "my-custom-output-v1"),
],
```

### How it works

Registration is lazy. Nothing is imported until `ModelManager.load()` is called. At that point:

1. Backend loads the model (`AutoModel.from_pretrained` for direct, worker subprocess for subprocess)
2. For direct backend: `lazy_register(type(model))` walks the class MRO
3. For subprocess backend: worker sends MRO class names in READY pipe, `lazy_register_by_names(mro_names)` matches by string
4. For each ancestor, checks if `cls.__name__` has an entry in `_TASK_CONFIGS`
5. If found, registers the tasks (imports only validators/serializers — pure Python, no heavy deps)
6. Subsequent `process()` calls use the registered entry for dispatch + serialization
