<a href="https://github.com/Ahmednull/L2CS-Net" target="_blank">L2CS-Net</a> is a gaze estimation model.

You can detect the direction in which someone is looking using the L2CS-Net model.

## Execution Modes

L2CS-Net gaze detection supports both local and remote execution modes when used in workflows:

- **Local execution**: The model runs directly on your inference server
- **Remote execution**: The model can be invoked via HTTP API on a remote inference server using `detect_gazes()` client method

## How to Use L2CS-Net

To use L2CS-Net with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, <a href="https://app.roboflow.com" target="_blank">sign up for a free Roboflow account</a>. Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

L2CS-Net accepts an image and returns pitch and yaw values that you can use to:

1. Figure out the direction in which someone is looking, and;
2. Estimate, roughly, where someone is looking.

We recommend using L2CS-Net paired with inference HTTP API. It's easy to set up with our `inference-cli` tool. Run the 
following command to set up environment and run the API under `http://localhost:9001`

```bash
pip install inference inference-cli inference-sdk
inference server start  # this starts server under http://localhost:9001
```


```python
import os
from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="http://localhost:9001",  # only local hosting supported
    api_key=os.environ["ROBOFLOW_API_KEY"]
)

CLIENT.detect_gazes(inference_input="./image.jpg")  # single image request
```

Above, replace `image.jpg` with the image in which you want to detect gazes.

The code above makes two assumptions:

1. Faces are roughly one meter away from the camera.
2. Faces are roughly 250mm tall.

These assumptions are a good starting point if you are using a computer webcam with L2CS-Net, where people in the frame are likely to be sitting at a desk.

On the first run, the model will be downloaded. On subsequent runs, the model will be cached locally and loaded from the cache. It will take a few moments for the model to download.

The results of L2CS-Net will appear in your terminal:

```
[{'face': {'x': 1107.0, 'y': 1695.5, 'width': 1056.0, 'height': 1055.0, 'confidence': 0.9355756640434265, 'class': 'face', 'class_confidence': None, 'class_id': 0, 'tracker_id': None, 'landmarks': [{'x': 902.0, 'y': 1441.0}, {'x': 1350.0, 'y': 1449.0}, {'x': 1137.0, 'y': 1692.0}, {'x': 1124.0, 'y': 1915.0}, {'x': 625.0, 'y': 1551.0}, {'x': 1565.0, 'y': 1571.0}]}, 'yaw': -0.04104889929294586, 'pitch': 0.029525401070713997}]
```

We have created a <a href="https://github.com/roboflow/inference/tree/main/examples/gaze-detection" target="_blank">full gaze detection example</a> that shows how to:

1. Use L2CS-Net with a webcam;
2. Calculate the direction in which and point in space at which someone is looking;
3. Calculate what quadrant of the screen someone is looking at, and;
4. Annotate the image with the direction someone is looking.

This example will let you run L2CS-Net and see the results of the model in real time. Here is an recording of the example working:

<video width="100%" autoplay loop muted>
  <source src="https://blog.roboflow.com/content/media/2023/09/gaze.mp4" type="video/mp4">
</video>

<a href="https://github.com/roboflow/inference/tree/main/examples/gaze-detection" target="_blank">Learn how to set up the example</a>.

## L2CS-Net Inference Response

Here is the structure of the data returned by a gaze request:

```python
[{'face': {'class': 'face',
           'class_confidence': None,
           'class_id': 0,
           'confidence': 0.9355756640434265,
           'height': 1055.0,
           'landmarks': [{'x': 902.0, 'y': 1441.0},
                         {'x': 1350.0, 'y': 1449.0},
                         {'x': 1137.0, 'y': 1692.0},
                         {'x': 1124.0, 'y': 1915.0},
                         {'x': 625.0, 'y': 1551.0},
                         {'x': 1565.0, 'y': 1571.0}],
           'tracker_id': None,
           'width': 1056.0,
           'x': 1107.0,
           'y': 1695.5},
  'pitch': 0.029525401070713997,
  'yaw': -0.04104889929294586}]
```

## See Also

- <a href="https://blog.roboflow.com/gaze-direction-position/" target="_blank">Gaze Detection and Eye Tracking: A How-To Guide</a>
