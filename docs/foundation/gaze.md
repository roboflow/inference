[L2CS-Net](https://github.com/Ahmednull/L2CS-Net) is a gaze estimation model.

You can detect the direction in which someone is looking using the L2CS-Net model.

## How to Use L2CS-Net

To use L2CS-Net with Inference, you will need a Roboflow API key. If you don't already have a Roboflow account, [sign up for a free Roboflow account](https://app.roboflow.com). Then, retrieve your API key from the Roboflow dashboard. Run the following command to set your API key in your coding environment:

```
export API_KEY=<your api key>
```

L2CS-Net accepts an image and returns pitch and yaw values that you can use to:

1. Figure out the direction in which someone is looking, and;
2. Estimate, roughly, where someone is looking.

Create a new Python file and add the following code:

```python
import base64

import cv2
import numpy as np
import requests
import os

IMG_PATH = "image.jpg"
API_KEY = os.environ["API_KEY"]
DISTANCE_TO_OBJECT = 1000  # mm
HEIGHT_OF_HUMAN_FACE = 250  # mm
GAZE_DETECTION_URL = "http://127.0.0.1:9001/gaze/gaze_detection?api_key=" + API_KEY

def detect_gazes(frame: np.ndarray):
    img_encode = cv2.imencode(".jpg", frame)[1]
    img_base64 = base64.b64encode(img_encode)
    resp = requests.post(
        GAZE_DETECTION_URL,
        json={
            "api_key": API_KEY,
            "image": {"type": "base64", "value": img_base64.decode("utf-8")},
        },
    )
    # print(resp.json())
    gazes = resp.json()[0]["predictions"]
    return gazes

image = cv2.imread(IMG_PATH)
gazes = detect_gazes(image)
print(gazes)
```

Above, replace `image.jpg` with the image in which you want to detect gazes.

The code above makes two assumptions:

1. Faces are roughly one meter away from the camera.
2. Faces are roughly 250mm tall.

These assumptions are a good starting point if you are using a computer webcam with L2CS-Net, where people in the frame are likely to be sitting at a desk.

Then, run the Python script you have created:

```
python app.py
```

On the first run, the model will be downloaded. On subsequent runs, the model will be cached locally and loaded from the cache. It will take a few moments for the model to download.

The results of L2CS-Net will appear in your terminal:

```
[{'face': {'x': 1107.0, 'y': 1695.5, 'width': 1056.0, 'height': 1055.0, 'confidence': 0.9355756640434265, 'class': 'face', 'class_confidence': None, 'class_id': 0, 'tracker_id': None, 'landmarks': [{'x': 902.0, 'y': 1441.0}, {'x': 1350.0, 'y': 1449.0}, {'x': 1137.0, 'y': 1692.0}, {'x': 1124.0, 'y': 1915.0}, {'x': 625.0, 'y': 1551.0}, {'x': 1565.0, 'y': 1571.0}]}, 'yaw': -0.04104889929294586, 'pitch': 0.029525401070713997}]
```

We have created a [full gaze detection example](https://github.com/roboflow/inference/tree/main/examples/gaze-detection) that shows how to:

1. Use L2CS-Net with a webcam;
2. Calculate the direction in which and point in space at which someone is looking;
3. Calculate what quadrant of the screen someone is looking at, and;
3. Annotate the image with the direction someone is looking.

This example will let you run L2CS-Net and see the results of the model in real time. Here is an recording of the example working:

<video width="100%" autoplay loop muted>
  <source src="https://blog.roboflow.com/content/media/2023/09/gaze.mp4" type="video/mp4">
</video>

[Learn how to set up the example](https://github.com/roboflow/inference/tree/main/examples/gaze-detection).

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

- [Gaze Detection and Eye Tracking: A How-To Guide](https://blog.roboflow.com/gaze-direction-position/)