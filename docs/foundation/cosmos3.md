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

If you only need to call the hosted Serverless API, install the lightweight SDK:

```pip install inference-sdk```

To run Cosmos 3 Edge locally, you need the `inference-gpu` package (the model requires a CUDA GPU) plus Cosmos-specific dependencies:

```
pip install inference-gpu
pip install -r https://raw.githubusercontent.com/roboflow/inference/main/requirements/requirements.cosmos.txt
```

!!! warning "Local installation requires pre-release dependencies"

    Cosmos 3 Edge support is newer than any released `transformers` version — the
    `cosmos3_edge` model type requires `transformers>=5.15`, which has not been
    published yet. The standard `inference-gpu[transformers]` extra installs an
    older release that cannot load the model, so the git-pinned versions in
    `requirements/requirements.cosmos.txt` are required until the upstream
    releases land. If you cannot install git-pinned dependencies, use the hosted
    Serverless API instead.

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

If you are running your own GPU-backed inference server (with the Cosmos dependencies from the Installation section above), point the same SDK code at it — for example a server running locally on port 9001:

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
