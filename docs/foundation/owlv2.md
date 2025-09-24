
<a href="https://arxiv.org/abs/2306.09683" target="_blank">OWLv2</a> is an open set object detectio model trained by Google. OWLv2 was primarily trained to detect objects from text. The implementation in `Inference` currently only supports detecting objects from visual examples of that object.

### Installation

To install inference with the extra dependencies necessary to run OWLv2, run

```pip install inference[transformers]```

or

```pip install inference-gpu[transformers]```

### How to Use OWLv2 

Create a new Python file called `app.py` and add the following code:

```python
import inference
from inference.models.owlv2.owlv2 import OwlV2
from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
from PIL import Image
import io
import base64

model = OwlV2()


im_url = "https://media.roboflow.com/inference/seawithdock.jpeg"
image = {
    "type": "url",
    "value": im_url
}
request = OwlV2InferenceRequest(
    image=image,
    training_data=[
        {
            "image": image,
            "boxes": [{"x": 223, "y": 306, "w": 40, "h": 226, "cls": "post"}],
        }
    ],
    visualize_predictions=True,
    confidence=0.9999,
)

response = OwlV2().infer_from_request(request)

def load_image_from_base64(base64_str):
    image = Image.open(io.BytesIO(base64_str))
    return image

visualization = load_image_from_base64(response.visualization)
visualization.save("owlv2_visualization.jpg")
```

In this code, we run OWLv2 on an image, using example objects from that image. Above, replace:

1. `training_data` with the locations of the objects you want to detect.
2. `im_url` with the image you would like to perform inference on.

Then, run the Python script you have created:

```
python app.py
```

The result from your model will be save to disk at `owlv2_visualization.jpg`

Note the blue bounding boxes surrounding each pole of the dock.


![OWLv2 results](https://media.roboflow.com/inference/owlv2_visualization.jpg)
