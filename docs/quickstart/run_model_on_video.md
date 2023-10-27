You can run Inference on video frames from `.mp4` and `.mov` files.

!!! tip "Tip"
    Follow our [Run a Fine-Tuned Model on Images](/docs/quickstart/run_model_on_image) guide to learn how to find a model to run.

## Run a Vision Model on a Video

To use fine-tuned with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export API_KEY=<your api key>
```

Create a new Python file and add the following code:

```python
import pickle
import cv2
import requests
import supervision as sv
import os

VIDEO_PATH = "football.mp4"
CLASS_LIST = ["box"]
MODEL_ID = "box-example"
CONFIDENCE = 0.5
API_KEY = os.environ["API_KEY"]

url = f"http://localhost:9001/{MODEL_ID}?image_type=numpy"
headers = {"Content-Type": "application/json"}
params = {
    "api_key": API_KEY,
    "confidence": confidence,
}

box_annotator = sv.BoxAnnotator()

for frame in sv.get_video_frames_generator(source_path=video_path):
    numpy_data = pickle.dumps(frame)
    response = requests.post(
        url, headers=headers, params=params, data=numpy_data
    ).json()
    detections = sv.Detections.from_roboflow(response, class_list=class_list)
    labels = [
        f"{class_list[class_id]} {confidence_value:0.2f}"
        for _, _, confidence_value, class_id, _ in detections
    ]
    annotated_image = box_annotator.annotate(
        frame, detections=detections, labels=labels
    )
    cv2.imshow("Annotated image", annotated_image)
    cv2.waitKey(1)
```

Above, set:

1. `VIDEO_PATH` to the path of your video.
2. `CLASS_LIST` to the list of classes you want to detect (these must be supported by the model you are using).
3. `MODEL_ID` to the ID of your model. Learn how to retrieve your model ID.
4. `CONFIDENCE` to the confidence threshold you want to use.
5. `API_KEY` to your API key. Learn how to retrieve your API key.

Then, run the Python script:

```
python app.py
```

Here is an example of inference run on a video of a football game:

<video width="100%" autoplay loop muted>
  <source src="https://media.roboflow.com/football-video.mp4" type="video/mp4">
</video>