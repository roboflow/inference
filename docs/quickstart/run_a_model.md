Let's run a computer vision model with Inference. The quickest way to get started with Inference is to simply load a model, and then call the model's `infer(...)` method.

## Install Inference

First, we need to install Inference:

{% include 'install.md' %}

To help us visualize our results in the example below, we will install Supervision:

```
pip install supervision
```

Create a new Python file called `app.py` and add the following code:

## Load a Model and Run Inference

```python
# import a utility function for loading Roboflow models
from inference import get_model

# define the image url to use for inference
image = "https://media.roboflow.com/inference/people-walking.jpg"

# load a pre-trained yolov8n model
model = get_model(model_id="yolov8n-640")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)
```

In the code above, we loaded a model and then we used that model's `infer(...)` method to run an image through it.

!!! tip

    When you run inference on an image, the same augmentations you applied when you generated a version in Roboflow will be applied at inference time. This helps improve model performance.

## Visualize Results

Running inference is fun but it's not much to look at. Let's add some code to visualize our results.

```python
from io import BytesIO

import requests
import supervision as sv
from inference import get_model
from PIL import Image
from PIL.ImageFile import ImageFile


def load_image_from_url(url: str) -> ImageFile:
    response = requests.get(url)
    response.raise_for_status()  # check if the request was successful
    image = Image.open(BytesIO(response.content))
    return image


# load the image from an url
image = load_image_from_url("https://media.roboflow.com/inference/people-walking.jpg")

# load a pre-trained yolov8n model
model = get_model(model_id="yolov8n-640")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0]

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results)

# create supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)
```

![People Walking Annotated](https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking-annotated.jpg)

## Summary

Huzzah! We used Inference to load a computer vision model, run inference on an image, then visualize the results! But this is just the start. There are many different ways to use Inference and how you use it is likely to depend on your specific use case and deployment environment. [Learn more about how to use inference here](../quickstart/inference_101.md).
