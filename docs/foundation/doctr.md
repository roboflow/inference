[DocTR](https://github.com/mindee/doctr) is an Optical Character Recognition (OCR) model.

You can use DocTR to read the text in an image.

To use DocTR with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). 

Then, retrieve your API key from the Roboflow dashboard. [Learn how to retrieve your API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).

Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

Let's retrieve the text in the following image:

![A shipping container](https://lh7-us.googleusercontent.com/rBXP1ngqRAfez18KyFjSPHX5Keo_hgb3La72sV5npNTf_Te63_pSSdpUnq_OeD5teh9RFg17yftljNSCuyURdNRRstKMtq-eolVEHhQF0XwnVgyqq6vaj4WbrNa0VUXmBic89jlJbHDnTUT4sT1i-bw)

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