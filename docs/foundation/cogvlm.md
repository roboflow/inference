[CogVLM](https://github.com/THUDM/CogVLM) is a Large Multimodal Model (LMM). CogVLM is available for use in Inference.

You can ask CogVLM questions about the contents of an image and retrieve a text response.

## Model Quantization

You can run CogVLM through Roboflow Inference with three degrees of quantization. Quantization allows you to make a model smaller, but there is an accuracy trade-off. The three degrees of quantization are:

- **No quantization**: Run the full model. For this, you will need 80 GB of RAM. You could run the model on an 80 GB NVIDIA A100.
- **8-bit quantization**: Run the model with less accuracy than no quantization. You will need 32 GB of RAM.You could run this model on an A100 with sufficient virtual RAM.
- **4-bit quantization**: Run the model with less accuracy than 8-bit quantization. You will need 16 GB of RAM. You could run this model on an NVIDIA T4.

## Use CogVLM with Inference

To use CogVLM with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). 

Then, retrieve your API key from the Roboflow dashboard. [Learn how to retrieve your API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

We recommend using CogVLM paired with inference HTTP API adjusted to run in GPU environment. It's easy to set up 
with our `inference-cli` tool. Run the following command to set up environment and run the API under 
`http://localhost:9001`

```bash
pip install inference inference-cli inference-sdk
inference server start  # make sure that you are running this at machine with GPU! Otherwise CogVLM will not be available
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

## See Also

- [How to deploy CogVLM](https://blog.roboflow.com/how-to-deploy-cogvlm/)