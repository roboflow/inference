Roboflow Inference is a powerful, performant, and easy to use inference package for computer vision and multi-modal machine learning models. With Inference, you spend less time reinventing the wheel and are able to quickly bring the power of computer vision into your software.

Before Inference, deploying models involved:

1. Writing custom inference logic, which often requires machine learning knowledge.
2. Managing dependencies.
3. Optimizing for performance and memory usage.
4. Writing tests to ensure your inference logic worked.
5. Writing custom interfaces to run your model over images and streams.

Inference handles all of this, out of the box.

## Hello Inference

First, install Inference via pip:

```bash
pip install inference
```

Then, use it to infer on an image with a public computer vision model:

```python
from inference import get_roboflow_model

model = get_roboflow_model(model_id="yolov8x-1280")

results = model.infer("people-walking.jpg")
```

<img width="100%" alt="people walking annotated" src="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking-annotated.jpg">

Let's break our example down step by step:

#### `from inference import get_roboflow_model`

First, we import a utility function which will help us load a computer vision model from Roboflow.

#### `model = get_roboflow_model(model_id="yolov8x-1280")`

Next, we load a model by referencing its `model_id`. For Roboflow models, the model ID is a combination of a project name and a version number `f"{project_name}/{version_number}"`. You can find your models project name and version number [in the Roboflow App](https://docs.roboflow.com/api-reference/workspace-and-project-ids). You can also browse public models that are ready to use on [Roboflow Universe](https://universe.roboflow.com/). In this example, we are using a special model ID that is an alias of [a COCO pretrained model on Roboflow Universe](https://universe.roboflow.com/microsoft/coco/model/13). You can see the list of model aliases [here](../../reference_pages/model_aliases).

#### `results = model.infer("people-walking.jpg")`

Finally, we run inference on a local image file ([hosted here](https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.jpg)). We used [Supervision](https://supervision.roboflow.com/how_to/detect_and_annotate/) to visualize the results.

## Why Inference?

With a single pip install you can start using Inference to run a fine-tuned model on any image or video stream. Inference supports running object detection, classification, instance segmentation, keypoint detection, and even foundation models (like CLIP and SAM). You can train and deploy your own custom model or use one of the [50,000+ fine-tuned models shared by the community](https://universe.roboflow.com).

With Inference, you spend more time building and less time integrating.
