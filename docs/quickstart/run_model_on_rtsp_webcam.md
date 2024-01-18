You can run computer vision models on webcam stream frames, RTSP stream frames, and video frames with Inference.

!!! tip "Follow our [Run a Fine-Tuned Model on Images](/quickstart/run_model_on_image) guide to learn how to find a model to run."

## Installation

To use fine-tuned models with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>. Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>

```

[Learn more about using Roboflow API keys in Inference](/quickstart/configure_api_key/)

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

Predictions are annotated using the `render_boxes` helper function. You can specify any function to process each prediction in the `on_prediction` parameter. [See our guide on creating custom sinks.](/quickstart/create_a_custom_inference_pipeline_sink/)

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

Presto! We used an InferencePipeline to run inference on our webcam and learned how we could modify it to run on other video sources (like video files or RTSP streams). See the [Inference Pipeline docs](/using_inference/inference_pipeline/) to learn more about other configurable parameters and built in sinks.
