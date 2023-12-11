[CogVLM](https://github.com/THUDM/CogVLM) is a Large Multimodal Model (LMM). CogVLM is available for use in Inference.

You can ask CogVLM questions about the contents of an image and retrieve a text response.

To use CogVLM with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). 

Then, retrieve your API key from the Roboflow dashboard. [Learn how to retrieve your API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

Run the following command to set your API key in your development environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

Let's ask a question about the following image:

[image]

Create a new Python file and add the following code:

```python
import requests
import base64
from PIL import Image
import os
from io import BytesIO

API_KEY = os.environ["API_KEY"]
IMAGE = "container.jpeg"

image = Image.open(IMAGE)
buffered = BytesIO()

image.save(buffered, quality=100, format="JPEG")

img_str = base64.b64encode(buffered.getvalue())
img_str = img_str.decode("ascii")

data = {
    "image": {
        "type": "base64",
        "value": img_str,
    }
}

ocr_results = requests.post("http://localhost:9001/doctr/ocr?api_key=" + API_KEY, json=data).json()

print(ocr_results)
```

Above, replace `container.jpeg` with the path to the image in which you want to detect objects.

Then, run the Python script you have created:

```
python app.py
```

The results of DocTR will appear in your terminal:

```
{'result': '', 'time': 3.98263641900121, 'result': 'MSKU 0439215', 'time': 3.870879542999319}
```

## See Also

- [How to detect text in images with OCR](https://blog.roboflow.com/ocr-api/)