[DocTR](https://github.com/mindee/doctr) is an Optical Character Recognition model.

You can use DocTR with Inference to identify and recognize characters in images.

### How to Use DocTR

To use DocTR with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export API_KEY=<your api key>
```

Create a new Python file and add the following code:

```python
import requests
import base64
from PIL import Image
import supervision as sv
import os

API_KEY = os.environ["API_KEY"]
IMAGE = "container1.jpeg"

image = Image.open(IMAGE)

data = {
    "image": {
        "type": "base64",
        "value": base64.b64encode(image.tobytes()).decode("utf-8"),
    }
}

ocr_results = requests.post("http://localhost:9001/doctr/ocr?api_key=" + API_KEY, json=data).json()

print(ocr_results, class_name)
```

Above, replace `container1.jpeg` with the path to the image in which you want to detect objects.

Then, run the Python script you have created:

```
python app.py
```

The results of DocTR will appear in your terminal:

```
...
```

## Further Reading

- [Use DocTR as part of a two-step detection system](https://blog.roboflow.com/ocr-api/)