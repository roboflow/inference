<a href="https://github.com/vikhyat/moondream" target="_blank">Moondream2</a> is a multimodal model that supports image captioning, zero-shot object detection, point-prompt detection, and visual question answering.

You can deploy Moondream2 with Inference.

### Execution Modes

Moondream2 supports both local and remote execution modes when used in workflows:

- **Local execution**: The model runs directly on your inference server (GPU recommended)
- **Remote execution**: The model can be invoked via HTTP API on a remote inference server using the `infer_lmm()` client method

### Installation

To install inference with the extra dependencies necessary to run Moondream2, run

```pip install inference[transformers]```

or

```pip install inference-gpu[transformers]```

### How to Use Moondream2

Create a new Python file called `app.py` and add the following code:

```python
from PIL import Image

from inference.models.moondream2.moondream2 import Moondream2

pg = Moondream2(api_key="API_KEY")

image = Image.open("dog.jpeg")

prompt = "How many dogs are in this image?"

result = pg.query(image, prompt)

print(result)
```

In this code, we load Moondream2 run Moondream2 on an image, and annotate the image with the predictions from the model.

Above, replace:

1. `prompt` with the prompt for the model.
2. `image.jpeg` with the path to the image that you want to run inference on.

To use Moondream2 with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>.

Then, run the Python script you have created:

```
python app.py
```

The result from your model will be printed to the console.
