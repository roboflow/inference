The native python API is the most simple and involves accessing the base package APIs directly. Going this route, you will import Inference modules directly into your python code. You will load models, run inference, and handle the results all within your own logic. You will also need to manage the dependencies within your python environment. If you are creating a simple app or just testing, the native Python API is a great place to start.

Using the native python API centers on loading models, then calling their `infer(...)` method to get inference results.

## Quickstart

This example shows how to load a model, run inference, then display the results.

{% include 'install.md' %}

Next, import a model:

```python
from inference import get_roboflow_model

model = get_roboflow_model(model_id="yolov8x-1280")
```

The `get_roboflow_model` method is a utility function which will help us load a computer vision model from Roboflow. We load a model by referencing its `model_id`. For Roboflow models, the model ID is a combination of a project name and a version number `f"{project_name}/{version_number}"`.

!!! Hint

    You can find your models project name and version number [in the Roboflow App](https://docs.roboflow.com/api-reference/workspace-and-project-ids). You can also browse public models that are ready to use on [Roboflow Universe](https://universe.roboflow.com/). In this example, we are using a special model ID that is an alias of [a COCO pretrained model on Roboflow Universe](https://universe.roboflow.com/microsoft/coco/model/13). You can see the list of model aliases [here](../../reference_pages/model_aliases).

Next, we can run inference with our model by providing an input image:

```python
from inference import get_roboflow_model

model = get_roboflow_model(model_id="yolov8x-1280")

results = model.infer("people-walking.jpg") # replace with path to your image
```

The results object is an [inference response object](../../docs/reference/inference/core/entities/responses/inference/#inference.core.entities.responses.inference.ObjectDetectionInferenceResponse). It contains some meta data (e.g. processing time) as well as an array of the predictions. The type of response and its attributes will depend on the type of model. [See all of the Inference Response objects](../../docs/reference/inference/core/entities/responses/inference/).

Now, lets visualize the results using [Supervision](https://supervision.roboflow.com):

```python
from inference import get_roboflow_model
import supervision as sv
import cv2

#Load model
model = get_roboflow_model(model_id="yolov8x-1280")

#Load image with cv2
image = cv2.imread("people-walking.jpg")

#Run inference
results = model.infer(image)

#Load results into Supervision Detection API
detections = sv.Detections.from_roboflow(
    results[0].dict(by_alias=True, exclude_none=True)
)

#Create Supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

#Extract labels array from inference results
labels = [p.class_name for p in results[0].predictions]



#Apply results to image using Supervision annotators
annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

#Write annotated image to file or display image
cv2.imwrite("people-walking-annotated.jpg", annotated_image)
#or sv.plot_image(annotated_image)

```

<img width="100%" alt="people walking annotated" src="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking-annotated.jpg">

## Different Image Types

In the previous example, we saw that we can provide different image types to the `infer(...)` method. The `infer(...)` method accepts images in many forms including PIL images, OpenCV images (Numpy arrays), paths to local images, image URLs, and more. Under the hood, models use the `load_image(...)` method in the [`image_utils` module](../../docs/reference/inference/core/utils/image_utils/).

```python
from inference import get_roboflow_model

import cv2
from PIL import Image

model = get_roboflow_model(model_id="yolov8x-1280")

image_url = "https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.jpg"
local_image_file = "people-walking.jpg"
pil_image = Image.open(local_image_file)
numpy_image = cv2.imread(local_image_file)

results = model.infer(image_url)
#or     = model.infer(local_image_file)
#or     = model.infer(pil_image)
#or     = model.infer(numpy_image)
```

## Inference Parameters

The `infer(...)` method accepts [keyword arguments to set inference parameters](../../docs/reference/inference/core/models/object_detection_base/#inference.core.models.object_detection_base.ObjectDetectionBaseOnnxRoboflowInferenceModel.infer). The example below shows setting the confidence threshold and the IoU threshold.

```python
from inference import get_roboflow_model

model = get_roboflow_model(model_id="yolov8x-1280")

results = model.infer("people-walking.jpg", confidence=0.75, iou_threshold=0.5)
```
