In Inference a _sink_ is a function used to execute logic on the inference results within an `InferencePipeline`. Inference has some built in sinks for convenience, but in this guide, we will learn to create a custom sink from scratch.

## Install Inference

{% include 'install.md' %}

## Setup An Inference Pipeline

Create a python file with the following code:

```python
# import the InferencePipeline interface
from inference import InferencePipeline

# create an inference pipeline object
pipeline = InferencePipeline.init(
    model_id="yolov8x-1280", # set the model id to a yolov8x model with in put size 1280
    video_reference="https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking.mp4", # set the video reference (source of video), it can be a link/path to a video file, an RTSP stream url, or an integer representing a device id (usually 0 for built in webcams)
    on_prediction=my_custom_sink, # THIS IS THE FUNCTION WE NEED TO CREATE
)

# start the pipeline
pipeline.start()
# wait for the pipeline to finish
pipeline.join()
```

In the code above, we have setup an Inference Pipeline with nearly everything it needs to run. We set the model ID to a pre-trained yolov8x model and we provided a link to a video as the video reference. The last remaining step is to define the `my_custom_sink` function which will be called after each inference is made.

## Define Custom Prediction Logic

The `on_prediction` parameter in the `InferencePipeline` constructor allows you to define custom prediction handlers. You can use this to define custom logic for how predictions are processed.

This function provides two parameters:

- `predictions`: A dictionary that contains all predictions returned by the model for the frame, and;
- `video_frame`: A [dataclass](../../docs/reference/inference/core/interfaces/camera/entities/#inference.core.interfaces.camera.entities.VideoFrame)

A VideoFrame object contains:

- `image`: The video frame as a NumPy array,
- `frame_id`: The frame ID, and;
- `frame_timestamp`: The timestamp of the frame.

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

# create a simple box annotator to use in our custom sink
annotator = sv.BoxAnnotator()

def my_custom_sink(predictions: dict, video_frame: VideoFrame):
    # get the text labels for each prediction
    labels = [p["class"] for p in predictions["predictions"]]
    # load our predictions into the Supervision Detections api
    detections = sv.Detections.from_inference(predictions)
    # annotate the frame using our supervision annotator, the video_frame, the predictions (as supervision Detections), and the prediction labels
    image = annotator.annotate(
        scene=video_frame.image.copy(), detections=detections, labels=labels
    )
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

And there you have it! We created a custom sink that takes the outputs of our Inference Pipeline, annotates an image, and displays it to our screen. See the [Inference Pipeline docs](/using_inference/inference_pipeline/) to learn more about other configurable parameters and built in sinks.
