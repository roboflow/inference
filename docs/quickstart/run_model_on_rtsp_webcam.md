You can run computer vision models on webcam or RTSP stream frames with Inference.

Webcam inference is ideal if you want to run a model on an edge device (i.e. an NVIDIA Jetson or Raspberry Pi).

RTSP inference is ideal for using models with internet connected cameras that support RTSP streaming.

You can run both fine-tuned models and foundation models. See the "Foundation Models" section in the sidebar to learn how to import and run foundation models.

!!! tip "Tip"
    Follow our [Run a Fine-Tuned Model on Images](/docs/quickstart/run_model_on_image) guide to learn how to find a model to run.

## Run a Vision Model on Webcam or RTSP Frames

To use fine-tuned with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export API_KEY=<your api key>
```

Once you have selected a model to run, create a new Python file and add the following code:

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
    source="webcam", # or "rstp://0.0.0.0:8000/password" for RTSP stream
    model="rock-paper-scissors-sxsw/11", # from Universe
    output_channel_order="BGR",
    use_main_thread=True, # for opencv display
    on_prediction=on_prediction, 
)
```

This code will run a model on frames from a webcam stream. To use RTSP, set the `source` value to an RTSP stream URL.

Predictions will be annotated using the [supervision Python package](https://github.com/roboflow/supervision).

Replace `rock-paper-scissors-sxsw/11` with the model ID associated with the mode you want to run.

Then, run the Python script:

```
python app.py
```

Your webcam will open and you can see the model running:

<video width="100%" autoplay loop muted>
  <source src="https://media.roboflow.com/rock-paper-scissors.mp4" type="video/mp4">
</video>