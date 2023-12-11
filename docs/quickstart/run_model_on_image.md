You can run fine-tuned models on images using Inference.

An Inference server will manage inference. Inference can be run on your local machine, a remote server, or even a Raspberry Pi.

If you need to deploy to the edge, you can use a device like the Jetson Nano. If you need high-performance compute for batch jobs, you can deploy Inference to a server with a GPU.

!!! tip "Follow our [Run a Fine-Tuned Model on Images](/docs/quickstart/run_model_on_image) guide to learn how to find a model to run."

!!! info
If you haven't already, follow our Run Your First Model guide to install and set up Inference.

Create a new Python file and add the following code:

```python
from inference.models.utils import get_roboflow_model
import numpy as np
from PIL import Image
import requests

image_url = (
    "https://storage.googleapis.com/com-roboflow-marketing/inference/soccer2.jpg"
)

model = get_roboflow_model(
    model_id="soccer-players-5fuqs/1",
    api_key="YOUR ROBOFLOW API KEY"
)

image = Image.open(
    requests.get(image_url, stream=True).raw
)  # load it as a PIL image so we can use it later for plotting
# or    Image.open("local/soccer/image.jpg") --> PIL Image
# or    cv2.imread("local/soccer/image.jpg") --> np.ndarray
# or    image_url  --> image url string

result = model.infer(image)

print(result)
```

Replace your API key, model ID, and model version as appropriate.

- [Learn how to find your API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)
- [Learn how to find your model ID](https://docs.roboflow.com/api-reference/workspace-and-project-ids)

!!! tip

    When you run inference on an image, the same augmentations you applied when you generated a version in Roboflow will be applied at inference time. This helps improve model performance.

Then, run the code. You will see predictions printed to the console as a list of inference result objects. Since we passed in a single image, the list will have length 1.

```
[ObjectDetectionInferenceResponse(visualization=None, frame_id=None, time=None, image=InferenceResponseImage(width=2304, height=1728), predictions=[ObjectDetectionPrediction(x=759.0, y=808.5, width=78.0, height=105.0, confidence=0.8841057419776917, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=968.5, y=837.0, width=63.0, height=156.0, confidence=0.8662749528884888, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=612.5, y=793.0, width=49.0, height=152.0, confidence=0.8658955097198486, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=1689.0, y=1146.5, width=144.0, height=141.0, confidence=0.8657780885696411, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=1622.5, y=951.0, width=79.0, height=160.0, confidence=0.8612774610519409, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=1408.5, y=910.5, width=59.0, height=159.0, confidence=0.8570612668991089, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=1061.5, y=833.0, width=67.0, height=136.0, confidence=0.854312539100647, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=1809.5, y=976.0, width=77.0, height=160.0, confidence=0.8437602519989014, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=2076.5, y=1308.0, width=115.0, height=184.0, confidence=0.8247343301773071, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=563.5, y=1180.0, width=69.0, height=182.0, confidence=0.8239980936050415, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=1510.5, y=820.0, width=77.0, height=140.0, confidence=0.8219611644744873, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=55.0, y=1115.0, width=84.0, height=208.0, confidence=0.8029934167861938, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=1911.0, y=1177.5, width=88.0, height=205.0, confidence=0.7846324443817139, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=1454.5, y=1224.5, width=121.0, height=207.0, confidence=0.7713653445243835, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=2255.5, y=1648.0, width=97.0, height=160.0, confidence=0.6980146169662476, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=1492.0, y=935.0, width=64.0, height=176.0, confidence=0.6881336569786072, class_name='player', class_confidence=None, class_id=1, tracker_id=None), ObjectDetectionPrediction(x=636.5, y=1037.0, width=55.0, height=182.0, confidence=0.6851024031639099, class_name='player', class_confidence=None, class_id=1, tracker_id=None)])]
```

You can plot predictions using `supervision`. You can install supervision using `pip install supervision`. Add the following code to the script you created to plot predictions from Inference:

```python
import supervision as sv

result = result[0].dict(
    by_alias=True, exclude_none=True
)  # convert the response object to a dictionary
detections = sv.Detections.from_roboflow(
    result
)  # convert the dictionary to a Supervision Detections object
labels = [
    p["class"] for p in result["predictions"]
]  # get the labels from the dictionary

box_annotator = sv.BoxAnnotator()  # create a box annotator
image = np.array(image)  # convert the PIL image to a numpy array
annotated_frame = box_annotator.annotate(
    scene=image.copy(), detections=detections, labels=labels
)  # annotate the image with the detections and labels

annotated_frame = Image.fromarray(
    annotated_frame
)  # convert the annotated image back to a PIL image
annotated_frame.save("annotated_frame.jpg")  # save the annotated image to disk
```
