<a href="https://github.com/THUDM/CogVLM" target="_blank">CogVLM</a> is a Large Multimodal Model (LMM). CogVLM is available for use in Inference.

You can ask CogVLM questions about the contents of an image and retrieve a text response.

!!! Warning "CogVLM reached **End Of Life**"

    Due to dependencies conflicts with newer models and security vulnerabilities discovered in `transformers`
    library patched in the versions of library incompatible with the model we announced End Of Life for CogVLM
    support in `inference`, effective since release `0.38.0`.

    We are leaving this page only for future reference, explicitly marking the last version of `inference` 
    supporting the feature (which is `0.37.1`). This tutorial should be treated as demonstration of 
    capabilities of Visual Language Models and **should not** be reproduced in any production enviromnets 
    (due to [security issues](https://nvd.nist.gov/vuln/detail/CVE-2024-11393)).

    We encourage Roboflow clients to try another Visual Language Models supported by `inference`, including 
    Qwen2.5-VL which is now available.

## Model Quantization

You can run CogVLM through Roboflow Inference with three degrees of quantization. Quantization allows you to make a model smaller, but there is an accuracy trade-off. The three degrees of quantization are:

- **No quantization**: Run the full model. For this, you will need 80 GB of RAM. You could run the model on an 80 GB NVIDIA A100.
- **8-bit quantization**: Run the model with less accuracy than no quantization. You will need 32 GB of RAM.You could run this model on an A100 with sufficient virtual RAM.
- **4-bit quantization**: Run the model with less accuracy than 8-bit quantization. You will need 16 GB of RAM. You could run this model on an NVIDIA T4.

## Use CogVLM with Inference

To use CogVLM with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>. 

Then, retrieve your API key from the Roboflow dashboard. <a href="https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key" target="_blank">Learn how to retrieve your API key</a>.

Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

We recommend using CogVLM paired with inference HTTP API adjusted to run in GPU environment. It's easy to set up 
with our `inference-cli` tool. Run the following command to set up environment and run the API under 
`http://localhost:9001`

!!! warning
    Make sure that you are running this at machine with an NVidia GPU! Otherwise CogVLM will not be available.


```bash
pip install "inference==0.37.1"
inference server start
```

Let's ask a question about the following image:

![A forklift in a warehouse](https://lh7-us.googleusercontent.com/4rgEU3nMJQzr54mYpGifEQp0hn3wu4oG8Sa21373M43eQ5TML-lBJyzYz3ZmPEETFwKnUGMmncsWA68wHo-4yzEGTV--TNCY7MJTxpJ-cS2w9JdUuIGVnwfAQN_72wK7TgGv-gtuLusJtAjAZxJVBFA)

Use `inference-sdk` to prompt the model:

```python
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # only local hosting supported
    api_key=os.environ["ROBOFLOW_API_KEY"]
)

result = CLIENT.prompt_cogvlm(
    visual_prompt="./forklift.jpg",
    text_prompt="Is there a forklift close to a conveyor belt?",
)
print(result)
```

Above, replace `forklift.jpeg` with the path to the image in which you want to detect objects.

Let's use the prompt "Is there a forklift close to a conveyor belt?”"

The results of CogVLM will appear in your terminal:

```python
{
    'response': 'yes, there is a forklift close to a conveyor belt, and it appears to be transporting a stack of items onto it.',
    'time': 12.89864671198302
}
```

CogVLM successfully answered our question, noting there is a forklift close to the conveyor belt in the image.

### Benchmarking

We ran 100 inferences on an NVIDIA T4 GPU to benchmark the performance of CogVLM.

CogVLM ran 100 inferences in 365.22 seconds (11.69 seconds per inference, on average).

## See Also

- <a href="https://blog.roboflow.com/how-to-deploy-cogvlm/" target="_blank">How to deploy CogVLM</a>