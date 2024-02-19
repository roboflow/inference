<a href="https://www.yoloworld.cc/" target="_blank">YOLO-World</a> is a zero-shot object detection model.

You can use YOLO-World to identify objects in images and videos using arbitrary text prompts.

To use YOLO-World effectively, we recommend experimenting with the model to understand which text prompts help achieve the desired results.

YOLO World is faster than many other zero-shot object detection models like YOLO-World. On powerful hardware like a V100 GPU, YOLO World can run in real-time.

!!! note

    YOLO-World, like most state-of-the-art zero-shot detection models, is most effective at identifying common objects (i.e. cars, people, dogs, etc.). It is less effective at identifying uncommon objects (i.e. a specific type of car, a specific person, a specific dog, etc.).

### How to Use YOLO-World

Create a new Python file called `app.py` and add the following code:

```python
import cv2
import supervision as sv

from inference.models.yolo_world.yolo_world import YOLOWorld

image = cv2.imread("image.jpeg")

model = YOLOWorld(model_id="yolo_world/l")
classes = ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"]
results = model.infer("image.jpeg", text=classes, confidence=0.03)

detections = sv.Detections.from_inference(results)

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

labels = [classes[class_id] for class_id in detections.class_id]

annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections
)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels
)

sv.plot_image(annotated_image)
```

In this code, we load YOLO-World, run YOLO-World on an image, and annotate the image with the predictions from the model.

Above, replace:

1. `["person", "backpack", "dog", "eye", "nose", "ear", "tongue"]` with the objects you want to detect.
2. `image.jpeg` with the path to the image in which you want to detect objects.

Then, run the Python script you have created:

```
python app.py
```

The result from YOLO-World will be displayed in a new window.

![YOLO-World results](https://media.roboflow.com/yolo-world-dog.png)