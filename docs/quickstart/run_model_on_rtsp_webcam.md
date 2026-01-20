You can run computer vision models on webcam stream frames, RTSP stream frames, and video frames with Inference.

!!! tip "Follow our [Run a Fine-Tuned Model on Images](/quickstart/run_model_on_image.md) guide to learn how to find a model to run."

## Installation

To use fine-tuned models with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>. Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>

```

[Learn more about using Roboflow API keys in Inference](../quickstart/configure_api_key.md)

Then, install Inference:

{% include 'install.md' %}

## Inference on Video

Next, create an Inference Pipeline. Once you have selected a model to run, create a new Python file and add the following code:

```python
# Import the InferencePipeline object
from inference import InferencePipeline
# Import the built in render_boxes sink for visualizing results
from inference.core.interfaces.stream.sinks import render_boxes

# initialize a pipeline object
pipeline = InferencePipeline.init(
    model_id="rock-paper-scissors-sxsw/11", # Roboflow model to use
    video_reference=0, # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
    on_prediction=render_boxes, # Function to run after each prediction
)
pipeline.start()
pipeline.join()
```

This code will run a model on frames from a webcam stream. To use RTSP, set the `video_reference` value to an RTSP stream URL. To use video, set the `video_reference` value to a video file path.

Predictions are annotated using the `render_boxes` helper function. You can specify any function to process each prediction in the `on_prediction` parameter.

Replace `rock-paper-scissors-sxsw/11` with the model ID associated with the model you want to run.

{% include 'model_id.md' %}

Then, run the Python script:

```bash
python app.py
```

Your webcam will open and you can see the model running:

<video width="100%" autoplay loop muted>
  <source src="https://media.roboflow.com/rock-paper-scissors.mp4" type="video/mp4">
</video>

!!! tip

    When you run inference on an image, the same augmentations you applied when you generated a version in Roboflow will be applied at inference time. This helps improve model performance.

Presto! We used an InferencePipeline to run inference on our webcam and learned how we could modify it to run on other video sources (like video files or RTSP streams). See the [Inference Pipeline docs](../using_inference/inference_pipeline.md) to learn more about other configurable parameters and built in sinks.

## Define Custom Prediction Logic

In Inference a _sink_ is a function used to execute logic on the inference results within an `InferencePipeline`. Inference has some built in sinks for convenience. We used one above to plot bounding boxes.

Below, we describe how to define custom prediction logic.

The `on_prediction` parameter in the `InferencePipeline` constructor allows you to define custom prediction handlers. You can use this to define custom logic for how predictions are processed.

This function provides two parameters:

- `predictions`: A dictionary that contains all predictions returned by the model for the frame, and;
- `video_frame`: A [dataclass](../../reference/inference/core/interfaces/camera/entities/#inference.core.interfaces.camera.entities.VideoFrame)

A VideoFrame object contains:

- `image`: The video frame as a NumPy array,
- `frame_id`: The frame ID, and;
- `frame_timestamp`: The timestamp of the frame.
- `source_id`: The index of the video_reference element which was passed to InferencePipeline (useful when multiple streams are passed to InferencePipeline).

Let's start by just printing the frame ID to the console.

```python
from inference import InferencePipeline
# import VideoFrame for type hinting
from inference.core.interfaces.camera.entities import VideoFrame

# define sink function
def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # print the frame ID of the video_frame object
    print(f"Frame ID: {video_frame.frame_id}")

pipeline = InferencePipeline.init(
    model_id="yolov8x-1280",
    video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
    on_prediction=my_custom_sink,
)

pipeline.start()
pipeline.join()
```

The output should look something like:

```bash
Frame ID: 1
Frame ID: 2
Frame ID: 3
```

Now let's do something a little more useful and use our custom sink to visualize our predictions.

```python
from inference import InferencePipeline
from inference.core.interfaces.camera.entities import VideoFrame

# import opencv to display our annotated images
import cv2
# import supervision to help visualize our predictions
import supervision as sv

# create a bounding box annotator and label annotator to use in our custom sink
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # get the text labels for each prediction
    labels = [p["class"] for p in predictions["predictions"]]
    # load our predictions into the Supervision Detections api
    detections = sv.Detections.from_inference(predictions)
    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
    image = label_annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
    image = box_annotator.annotate(image, detections=detections)
    # display the annotated image
    cv2.imshow("Predictions", image)
    cv2.waitKey(1)

pipeline = InferencePipeline.init(
    model_id="yolov8x-1280",
    video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4",
    on_prediction=my_custom_sink,
)

pipeline.start()
pipeline.join()
```

You should see something like this on your screen:

<video width="100%" autoplay loop muted>
  <source src="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking-annotated.mp4" type="video/mp4">
</video>

And there you have it! We created a custom sink that takes the outputs of our Inference Pipeline, annotates an image, and displays it to our screen. See the [Inference Pipeline docs](/using_inference/inference_pipeline) to learn more about other configurable parameters and built in sinks.

## Existing Video Sinks

### Built In Sinks

Inference has [several sinks built in](../../reference/inference/core/interfaces/stream/sinks/) that are ready to use.

#### `render_boxes(...)`

The [render boxes sink](../../reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.render_boxes) is made to visualize predictions and overlay them on a stream. It uses Supervision annotators to render the predictions and display the annotated frame.
It only works for Roboflow models that yields detection-based output (`object-detection`, `instance-segmentation`, `keypoint-detection`), yet not all details of predictions may be 
displayed by default (like detected key-points).

#### `UDPSink(...)`

The [UDP sink](../../reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.UDPSink) is made to broadcast predictions with a UDP port. This port can be listened to by client code for further processing.
It uses Python-default json serialisation - so predictions must be serializable, otherwise error will be thrown.  

#### `multi_sink(...)`

The [Multi-Sink](../../reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.multi_sink) is a way to combine multiple sinks so that multiple actions can happen on a single inference result.

#### `VideoFileSink(...)`

The [Video File Sink](../../reference/inference/core/interfaces/stream/sinks/#inference.core.interfaces.stream.sinks.VideoFileSink) visualizes predictions, similar to the `render_boxes(...)` sink, however, instead of displaying the annotated frames, it saves them to a video file.
All constraints related to `render_boxes(...)` apply.
