---
description: Reason about physical scenes with NVIDIA Cosmos 3 Edge, a vision-language world model for spatial understanding, safety, and prediction.
---

# Cosmos 3 Edge

<a href="https://huggingface.co/nvidia/Cosmos3-Edge" target="_blank">Cosmos 3 Edge</a> is a vision-language "world model" developed by NVIDIA and tuned for physical scene understanding.

Unlike a general-purpose captioning model, Cosmos 3 Edge is trained to reason about the physical world. You can use it to ask questions about spatial relations between objects, assess safety conditions in a scene, and predict what is likely to happen next. This makes it well suited to robotics, industrial monitoring, and other applications where understanding physical context matters more than a literal description.

Cosmos 3 Edge exposes a "thinking" mode: internally the model can generate reasoning tokens before it answers. By default, Inference strips this reasoning block and returns only the final answer.

You can deploy Cosmos 3 Edge with Inference.

### Model Details

| | |
|:--|:--|
| Model ID | `nvidia/cosmos-3-edge` |
| Task type | LMM / VLM (`lmm`) |
| License | OpenMDW-1.1 |

### Execution Modes

Cosmos 3 Edge supports both local and remote execution modes when used in workflows:

- **Local execution**: The model runs directly on your inference server. Cosmos 3 Edge requires an NVIDIA CUDA GPU — there is no CPU implementation, so local execution is not available on CPU-only hosts.
- **Remote execution**: The model can be invoked via HTTP API on a remote inference server using the `infer_lmm()` client method. The hosted Roboflow Serverless API supports Cosmos 3 Edge as well.

Cosmos 3 Edge is served exclusively through the `inference_models` backend (a Hugging Face transformers backend). It has no legacy implementation. When running your own server, this backend is enabled by default (`USE_INFERENCE_MODELS`), and the model endpoint is registered when `COSMOS3_ENABLED=True` (the default).

### Installation

Whether you call the hosted Serverless API or a local inference server, the only package you need to install is the lightweight SDK:

```
pip install inference-sdk
```

To serve Cosmos 3 Edge locally, run the dedicated Cosmos build of the GPU inference server Docker image — it ships with the Cosmos-specific dependencies pre-installed. The easiest way to start it is with the CLI:

```
pip install inference-cli
inference server start --image roboflow/roboflow-inference-server-gpu:1.3.8-cosmos3
```

This starts the server on port 9001 (an NVIDIA CUDA GPU is required — there is no CPU support for this model).

!!! warning "Local execution requires the `-cosmos3` GPU Docker image"

    Cosmos 3 Edge support is newer than any released `transformers` version — the
    `cosmos3_edge` model type requires a pre-release `transformers` build that
    cannot be shipped through the `inference-gpu` pip extras or the standard
    release images. The standard `roboflow/roboflow-inference-server-gpu:latest`
    image does NOT include these dependencies; only the `-cosmos3` suffixed tags
    (e.g. `1.3.8-cosmos3`) bundle the required git-pinned builds
    (`requirements/requirements.cosmos.txt`). If you cannot run the Docker
    image, use the hosted Serverless API instead.

### How to Use Cosmos 3 Edge (Hosted API)

The quickest way to try Cosmos 3 Edge is to call the hosted Roboflow Serverless API with the Inference SDK. Create a new Python file called `app.py` and add the following code:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="YOUR_ROBOFLOW_API_KEY",
)

result = client.infer_lmm(
    inference_input="https://media.roboflow.com/dog.jpeg",
    model_id="nvidia/cosmos-3-edge",
    prompt="Is the walkway free of obstacles?",
)

print(result["response"])
```

Above, replace:

1. `prompt` with the prompt for the model.
2. The image URL with the path to the image that you want to run inference on.

To use Cosmos 3 Edge with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>.

Then, run the Python script you have created:

```
python app.py
```

The model's text answer will be printed to the console.

### How to Use Cosmos 3 Edge (Local Inference Server)

If you are running the `-cosmos3` GPU inference server Docker image (see the Installation section above), point the same SDK code at it — for example a server running locally on port 9001:

```python
from inference_sdk import InferenceHTTPClient

client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="YOUR_ROBOFLOW_API_KEY",
)

result = client.infer_lmm(
    inference_input="https://media.roboflow.com/dog.jpeg",
    model_id="nvidia/cosmos-3-edge",
    prompt="Describe what's in this image.",
)

print(result["response"])
```

### How to Use Cosmos 3 Edge (Local Python)

You can also run the model in-process with the `inference` Python package, without going through the HTTP API. Because the Cosmos dependencies ship only in the `-cosmos3` Docker image, the easiest way is to run your script inside that image. Create `app.py`:

```python
from inference import get_model

model = get_model("nvidia/cosmos-3-edge", api_key="YOUR_ROBOFLOW_API_KEY")

result = model.infer(
    "https://media.roboflow.com/dog.jpeg",
    prompt="What is likely going to happen next in this scene?",
)

print(result[0].response)
```

Then run it inside the Cosmos image:

```bash
docker run --rm --gpus all \
  -v $(pwd):/workspace -w /workspace \
  -v /tmp/model-cache:/tmp/model-cache -e MODEL_CACHE_DIR=/tmp/model-cache \
  --entrypoint python3 \
  roboflow/roboflow-inference-server-gpu:1.3.8-cosmos3 app.py
```

The `-v /tmp/model-cache` mount persists the downloaded weights across runs (and is the same cache directory `inference server start` uses).

Under the hood the model is loaded through the `inference_models` package — you can equivalently load it directly with `AutoModel`:

```python
import cv2
from inference_models import AutoModel

model = AutoModel.from_pretrained("nvidia/cosmos-3-edge", api_key="YOUR_ROBOFLOW_API_KEY")

image = cv2.imread("my-image.jpg")
answers = model.prompt(images=image, prompt="What is likely going to happen next in this scene?")
print(answers[0])
```

### System Prompts

Cosmos 3 Edge accepts an optional system prompt that lets you steer the model's behavior — for example, asking it to answer as a safety inspector. When calling the model directly, the system prompt is appended to the user prompt with the `<system_prompt>` delimiter:

```python
result = client.infer_lmm(
    inference_input="https://media.roboflow.com/dog.jpeg",
    model_id="nvidia/cosmos-3-edge",
    prompt="Is this scene safe for a robot to enter?<system_prompt>You are a safety inspector.",
)
```

If you do not provide a prompt, Cosmos 3 Edge falls back to a default scene-description prompt (`Describe what's in this image.`), and a default system prompt that frames the model as an assistant that understands physical scenes.

### Use With Workflows

Cosmos 3 Edge is also available as a Workflows block. Add the **Cosmos 3** block (`roboflow_core/cosmos3_edge@v1`) to a workflow to run the model as a step in a larger pipeline.

The block accepts the following inputs:

- **images**: the image(s) to run inference on.
- **prompt** (optional): the text prompt to pass to the model. Defaults to `Describe what's in this image.`
- **system_prompt** (optional): an additional system prompt used to steer the model's behavior.

The `model_version` is fixed to `nvidia/cosmos-3-edge`. For each input image, the block produces an `output` string containing the model's text answer.
