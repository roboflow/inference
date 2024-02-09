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
from inference import get_roboflow_model

# define the image url to use for inference
image = "https://media.roboflow.com/inference/people-walking.jpg"

# load a pre-trained yolov8n model
model = get_roboflow_model(model_id="yolov8n-640")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)
```

In the code above, we loaded a model and then we used that model's `infer(...)` method to run an image through our computer vision model.

!!! tip

    When you run inference on an image, the same augmentations you applied when you generated a version in Roboflow will be applied at inference time. This helps improve model performance.

## Visualize Results

Running inference is fun but it's not much to look at. Let's add some code to visualize our results.

```python
# import a utility function for loading Roboflow models
from inference import get_roboflow_model
# import supervision to visualize our results
import supervision as sv
# import cv2 to helo load our image
import cv2

# define the image url to use for inference
image_file = "people-walking.jpg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_roboflow_model(model_id="yolov8n-640")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)
```

The `people-walking.jpg` file is hosted <a href="https://media.roboflow.com/inference/people-walking.jpg" target="_blank">here</a>.

![People Walking Annotated](https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking-annotated.jpg)

## Summary

Huzzah! We used Inference to load a computer vision model, run inference on an image, then visualize the results! But this is just the start. There are many different ways to use Inference and how you use it is likely to depend on your specific use case and deployment environment. [Learn more about how to use inference here](/quickstart/inference_101/).
