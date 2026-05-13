<a href="https://huggingface.co/zai-org/GLM-OCR" target="_blank">GLM-OCR</a> is a vision-language model by Zhipu AI (ZAI) for Optical Character Recognition (OCR).

GLM-OCR uses a modern image-text-to-text architecture for high-quality text recognition. It supports custom prompts to guide recognition for different use cases such as serial numbers, labels, and document text.

To use GLM-OCR with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>.

Then, retrieve your API key from the Roboflow dashboard. <a href="https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key" target="_blank">Learn how to retrieve your API key</a>.

Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

Let's try running GLM-OCR on this image:

![Serial number](https://media.roboflow.com/serial_number.png)

!!! note

    GLM-OCR requires `inference` with `inference-models` support (`USE_INFERENCE_MODELS=true`) and a GPU.

    To run the example, start `inference` server locally:

    ```bash
    pip install inference-cli && inference server start
    ```

### Using the Inference SDK

```python
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://127.0.0.1:9001",
    api_key=os.environ["ROBOFLOW_API_KEY"]
)

result = CLIENT.infer_lmm(
    inference_input="./serial_number.png",
    prompt="Text Recognition:",
    model_id="glm-ocr",
)
print(result["response"])
```

## See Also

- <a href="https://huggingface.co/zai-org/GLM-OCR" target="_blank">GLM-OCR on Hugging Face</a>
- <a href="https://blog.roboflow.com/ocr-api/" target="_blank">How to detect text in images with OCR</a>
