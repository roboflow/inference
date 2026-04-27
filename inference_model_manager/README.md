# inference-model-manager

Manages model lifecycle (load, unload, evict) and dispatches inference requests. Sits between your code and `inference-models` — you don't call models directly.

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
