Let's run a fine-tuned computer vision model to play rock paper scissors.

The Inference Server runs in Docker. Before we begin, make sure you have installed Docker on your system. To learn how to install Docker, refer to the [official Docker installation guide](https://docs.docker.com/get-docker/).

Next, we need to install Inference:

```
pip install inference
```

To help us visualize our results in the example below, we will install Supervision:

```
pip install supervision
```

Create a new Python file called `app.py` and add the following code:

```python
import cv2
import inference
import supervision as sv

annotator = sv.BoxAnnotator()

def on_prediction(predictions, image):
    labels = [p["class"] for p in predictions["predictions"]]
    detections = sv.Detections.from_roboflow(predictions)
    cv2.imshow(
        "Prediction",
        annotator.annotate(
            scene=image,
            detections=detections,
            labels=labels
        )
    ),
    cv2.waitKey(1)

inference.Stream(
    source="webcam", # or rtsp stream or camera id
    model="rock-paper-scissors-sxsw/11", # from Universe
    output_channel_order="BGR",
    use_main_thread=True, # for opencv display
    on_prediction=on_prediction,
)
```

Next, sign up for a [free Roboflow account](https://app.roboflow.com). Retrieve your API key from the Roboflow dashboard, then run the following command:

```
export ROBOFLOW_API_KEY=<your api key>
```

Then, run the Python script:

```
python app.py
```

Your webcam will open and you can play rock paper scissors:

<video width="100%" autoplay loop muted>
  <source src="https://media.roboflow.com/rock-paper-scissors.mp4" type="video/mp4">
</video>


!!! tip

    When you run inference on an image, the same augmentations you applied when you generated a version in Roboflow will be applied at inference time. This helps improve model performance.