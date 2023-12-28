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

Run the following command to set your API key in your development environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

Let's ask a question about the following image:

![A forklift in a warehouse](https://lh7-us.googleusercontent.com/4rgEU3nMJQzr54mYpGifEQp0hn3wu4oG8Sa21373M43eQ5TML-lBJyzYz3ZmPEETFwKnUGMmncsWA68wHo-4yzEGTV--TNCY7MJTxpJ-cS2w9JdUuIGVnwfAQN_72wK7TgGv-gtuLusJtAjAZxJVBFA)

Create a new Python file and add the following code:

```python
import base64
import os
from PIL import Image
import requests

PORT = 9001
API_KEY = os.environ["API_KEY"]
IMAGE_PATH = "forklift.png"


def encode_base64(image_path):
    with open(image_path, "rb") as image:
        x = image.read()
        image_string = base64.b64encode(x)

    return image_string.decode("ascii")

prompt = "Is there a forklift close to a conveyor belt?"

infer_payload = {
    "image": {
        "type": "base64",
        "value": encode_base64(IMAGE_PATH),
    },
    "api_key": API_KEY,
    "prompt": prompt,
}

results = requests.post(
    f"http://localhost:{PORT}/llm/cogvlm",
    json=infer_payload,
)

print(results.json())
```

Above, replace `forklift.jpeg` with the path to the image in which you want to detect objects.

Let's use the prompt "Is there a forklift close to a conveyor belt?”"

Run the Python script you have created:

```bash
python app.py
```

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