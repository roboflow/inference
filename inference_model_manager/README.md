# inference-model-manager

Manages model lifecycle (load, unload, evict) and dispatches inference requests. Sits between your code and `inference-models` — you don't call models directly.

## Install

Requires Python 3.10–3.13. From this directory:

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

## Additional backends

`backend=` also accepts any name registered via the
`inference_model_manager.backends` entry-point group. A plugin backend's
model does not have to live in-process — it implements `submit_request()`
instead of exposing `.model` directly (see
`inference_model_manager/backends/base.py` for the full contract).

## Color format

Models default to BGR input (OpenCV convention). Pass `input_color_format` if your source is different:

| Source | Format | Pass to `process()` |
|--------|--------|---------------------|
| `imagecodecs.imread()` | RGB | `input_color_format="rgb"` |
| `cv2.imread()` | BGR | nothing (default) |
| `PIL.Image` → `np.array()` | RGB | `input_color_format="rgb"` |
| torch tensor (CHW) | RGB | nothing (tensor default is RGB) |

## Registering a model in the registry

Models in `inference-models` work standalone — no changes needed there. To make a model available through model manager (action dispatch, validation, typed serialization), add an entry to `registry_defaults.py`.

### Case 1: Model inherits from a registered base class

If your model inherits from `ObjectDetectionModel`, `ClassificationModel`, `InstanceSegmentationModel`, etc. — **nothing to do**. The registry matches by class name via MRO. Your model inherits the base class entry automatically.

```python
# inference_models/models/my_detector/my_detector.py
class MyDetector(ObjectDetectionModel):
    def infer(self, images, **kwargs):
        ...
```

This works out of the box with `mm.process("my-detector", images=img)`.

### Case 2: New base class or model with unique actions

Add entries to `_ACTION_CONFIGS` in `registry_defaults.py`. Each entry is a tuple:

```
(action_name, method_name, is_default, params_dict, validator_name, serializer_name, response_type)
```

Example — a model with two actions:

```python
# In registry_defaults.py _ACTION_CONFIGS dict:
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
- **action_name** — what users pass as `action=` param (e.g. `mm.process("model", action="embed")`)
- **method_name** — actual method on the model class to call (can differ from action_name)
- **is_default** — exactly one action must be `True`; used when `action=None`
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

Then reference by name in `_ACTION_CONFIGS`:

```python
"MyCustomModel": [
    ("infer", "infer", True, ["images", "language"],
     "validate_my_custom_input", "serialize_my_custom_output",
     "my-custom-output-v1"),
],
```

### Case 4: Multi-model pipelines

A pipeline registered in `inference_models` (`REGISTERED_PIPELINES`) is loaded as one manager entry. `inference_model_manager/pipelines.py` maps the serving id family to the pipeline: `pp_ocr/{det}-{rec}` loads `pp-ocrv6-det/{det}` and `pp-ocrv6-rec/{rec}` (`none` disables a stage, a single token applies to both, bare `pp_ocr` uses the library defaults) and composes them with `resolve_pipeline_class(...).with_models(...)`. The composed object is wrapped in a facade whose class name carries the `_ACTION_CONFIGS` entry, so dispatch, validation and serialization work as for any model:

```python
mm.load("pp_ocr/small-small", api_key="YOUR_KEY")
texts, detections = mm.process("pp_ocr/small-small", images=image, serialize=False)
```

To add a pipeline: one `PipelineFamily` row in `PIPELINE_FAMILIES`, one facade adapting the pipeline result to a registered action contract, one `_ACTION_CONFIGS` entry keyed by the facade class name.

### How it works

Registration is lazy. Nothing is imported until `ModelManager.load()` is called. At that point:

1. Backend loads the model (`AutoModel.from_pretrained` for direct, backend-specific for plugin backends)
2. For direct backend: `lazy_register(type(model))` walks the class MRO
3. For backends whose model doesn't live in-process: the backend reports MRO class names directly, `lazy_register_by_names(mro_names)` matches by string
4. For each ancestor, checks if `cls.__name__` has an entry in `_ACTION_CONFIGS`
5. If found, registers the actions (imports only validators/serializers — pure Python, no heavy deps)
6. Subsequent `process()` calls use the registered entry for dispatch + serialization
