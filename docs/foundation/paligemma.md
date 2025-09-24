<a href="https://blog.roboflow.com/paligemma-multimodal-vision/" target="_blank">PaliGemma</a> is a large multimodal model developed by Google Research.

You can use PaliGemma to:

1. Ask questions about images (Visual Question Answering)
2. Identify the location of objects in an image (object detection)
3. Identify the precise location of objects in an imageh (image segmentation)

You can deploy PaliGemma object detection models with Inference, and use PaliGemma for object detection.

### Installation

To install inference with the extra dependencies necessary to run PaliGemma, run

```pip install inference[transformers]```

or

```pip install inference-gpu[transformers]```

### How to Use PaliGemma (VQA)

Create a new Python file called `app.py` and add the following code:

```python
import inference

from inference.models.paligemma.paligemma import PaliGemma

pg = PaliGemma("paligemma-3b-mix-224", api_key="YOUR ROBOFLOW API KEY")

from PIL import Image

image = Image.open("image.jpeg") # Change to your image

prompt = "How many dogs are in this image?"

result = pg.predict(image,prompt)
```

In this code, we load PaliGemma run PaliGemma on an image, and annotate the image with the predictions from the model.

Above, replace:

1. `prompt` with the prompt for the model.
2. `image.jpeg` with the path to the image in which you want to detect objects.

To use PaliGemma with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>.

Then, run the Python script you have created:

```
python app.py
```

The result from your model will be printed to the console.

### How to Use PaliGemma (Object Detection)

Create a new Python file called `app.py` and add the following code:

```python
import os
import transformers
import re
import numpy as np
import supervision as sv
from typing import Tuple, List, Optional
from PIL import Image

image = Image.open("/content/data/dog.jpeg")

def from_pali_gemma(response: str, resolution_wh: Tuple[int, int], class_list: Optional[List[str]] = None) -> sv.Detections:
    _SEGMENT_DETECT_RE = re.compile(
        r'(.*?)' +
        r'<loc(\d{4})>' * 4 + r'\s*' +
        '(?:%s)?' % (r'<seg(\d{3})>' * 16) +
        r'\s*([^;<>]+)? ?(?:; )?',
    )

    width, height = resolution_wh
    xyxy_list = []
    class_name_list = []

    while response:
        m = _SEGMENT_DETECT_RE.match(response)
        if not m:
            break

        gs = list(m.groups())
        before = gs.pop(0)
        name = gs.pop()
        y1, x1, y2, x2 = [int(x) / 1024 for x in gs[:4]]
        y1, x1, y2, x2 = map(round, (y1*height, x1*width, y2*height, x2*width))

        content = m.group()
        if before:
            response = response[len(before):]
            content = content[len(before):]

        xyxy_list.append([x1, y1, x2, y2])
        class_name_list.append(name.strip())
        response = response[len(content):]

    xyxy = np.array(xyxy_list)
    class_name = np.array(class_name_list)

    if class_list is None:
        class_id = None
    else:
        class_id = np.array([class_list.index(name) for name in class_name])

    return sv.Detections(
        xyxy=xyxy,
        class_id=class_id,
        data={'class_name': class_name}
    )

prompt = "detect person; car; backpack"
response = pali_gemma.predict(image, prompt)[0]
print(response)

detections = from_pali_gemma(response=response, resolution_wh=image.size, class_list=['person', 'car', 'backpack'])

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

annotatrd_image = bounding_box_annotator.annotate(image, detections)
annotatrd_image = label_annotator.annotate(annotatrd_image, detections)
sv.plot_image(annotatrd_image)
```

In this code, we load PaliGemma run PaliGemma on an image, and annotate the image with the predictions from the model.

Above, replace:

1. `prompt` with the prompt for the model.
2. `image.jpeg` with the path to the image in which you want to detect objects.

To use PaliGemma with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>.

Then, run the Python script you have created:

```
python app.py
```

The result from the model will be displayed:

![PaliGemma results](https://media.roboflow.com/inference/paligemma.png)
