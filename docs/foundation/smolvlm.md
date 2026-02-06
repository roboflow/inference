<a href="https://blog.roboflow.com/smolvlm2/" target="_blank">SmolVLM2</a> is a multimodal model developed by Hugging Face.

You can use SmolVLM2 for a range of multimodal tasks, including VQA, document OCR, document VQA, and object counting.

You can deploy SmolVLM2 with Inference.

### Execution Modes

SmolVLM2 supports both local and remote execution modes when used in workflows:

- **Local execution**: The model runs directly on your inference server (GPU recommended)
- **Remote execution**: The model can be invoked via HTTP API on a remote inference server using the `infer_lmm()` client method

### Installation

To install inference with the extra dependencies necessary to run SmolVLM2, run

```pip install inference[transformers]```

or

```pip install inference-gpu[transformers]```

### How to Use SmolVLM2

Create a new Python file called `app.py` and add the following code:

```python
from PIL import Image

from inference.models.smolvlm.smolvlm import SmolVLM

pg = SmolVLM(api_key="API_KEY")

image = Image.open("dog.jpeg")

prompt = "How many dogs are in this image?"

result = pg.predict(image,prompt)

print(result)
```

In this code, we load SmolVLM2 run SmolVLM2 on an image, and annotate the image with the predictions from the model.

Above, replace:

1. `prompt` with the prompt for the model.
2. `image.jpeg` with the path to the image that you want to run inference on.

To use SmolVLM2 with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>.

Then, run the Python script you have created:

```
python app.py
```

The result from your model will be printed to the console.
