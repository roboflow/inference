# Qwen 3.5

<a href="https://github.com/QwenLM/Qwen3.5" target="_blank">Qwen 3.5-VL</a> is a vision-language model developed by Alibaba.

You can use Qwen 3.5-VL for a range of multimodal tasks, including image understanding, visual question answering, and document analysis. It also supports a "thinking" mode that lets the model generate reasoning tokens before answering.

You can deploy Qwen 3.5-VL with Inference.

### Model Variants

Qwen 3.5-VL is available in two sizes:

| Model ID | Parameters |
|:---------|:-----------|
| `qwen3_5-0.8b` | 0.8B |
| `qwen3_5-2b` | 2B |

### Execution Modes

Qwen 3.5-VL supports both local and remote execution modes when used in workflows:

- **Local execution**: The model runs directly on your inference server (GPU recommended)
- **Remote execution**: The model can be invoked via HTTP API on a remote inference server

### Installation

To install inference with the extra dependencies necessary to run Qwen 3.5-VL, run

```pip install "inference[transformers]"```

or

```pip install "inference-gpu[transformers]"```

### How to Use Qwen 3.5-VL

Create a new Python file called `app.py` and add the following code:

```python
from inference.models.qwen3_5vl.qwen3_5vl_inference_models import (
    InferenceModelsQwen35VLAdapter,
)

model = InferenceModelsQwen35VLAdapter(
    model_id="qwen3_5-0.8b",
    api_key="YOUR_ROBOFLOW_API_KEY",
)

image = "https://media.roboflow.com/dog.jpeg"
prompt = "How many dogs are in this image?"

preprocessed, metadata = model.preprocess(image, prompt)
predictions = model.predict(preprocessed)
result = model.postprocess(predictions, metadata)

print(result[0].response)
```

Above, replace:

1. `prompt` with the prompt for the model.
2. The image URL with the path to the image that you want to run inference on.

To use Qwen 3.5-VL with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>.

Then, run the Python script you have created:

```
python app.py
```

The result from your model will be printed to the console.
