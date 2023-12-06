# HTTP Inference

The Roboflow Inference Server provides a standard API through which to run inference on computer vision models.

In this guide, we show how to run inference on object detection, classification, and segmentation models using the Inference Server.

Currently, the server is compatible with models trained on Roboflow, but stay tuned as we actively develop support for bringing your own models.

To run inference with the server, we will:

1. Install the server
2. Download a model for use on the server
3. Run inference

## Step #1: Install the Inference Server

_You can skip this step if you already have Inference installed and running._

The Inference Server runs in Docker. Before we begin, make sure you have installed Docker on your system. To learn how to install Docker, refer to the [official Docker installation guide](https://docs.docker.com/get-docker/).

Once you have Docker installed, you are ready to download Roboflow Inference. The command you need to run depends on what device you are using.

Start the server using `inference server start`. After you have installed the Inference Server, the Docker container will start running the server at `localhost:9001`.

## Step #2: Run Inference

You can send a URL with an image, a NumPy array, or a base64-encoded image to an Inference server. The server will return a JSON response with the predictions.

Choose an option below:

!!! Run Inference

    === "URL"
    
        Create a new Python file and add the following code:

        ```python
        import requests

        project_id = ""
        model_version = ""
        image_url = ""
        confidence = 0.75
        api_key = ""
        task = "object_detection"

        infer_payload = {
            "model_id": f"{project_id}/{model_version}",
            "image": {
                "type": "url",
                "value": image_url,
            },
            "confidence": confidence,
            "iou_threshold": iou_thresh,
            "api_key": api_key,
        }
        res = requests.post(
            f"http://localhost:9001/infer/{task}",
            json=infer_payload,
        )

        predictions = res.json()
        ```

        Above, specify:

        1. `project_id`, `model_version`: Your project ID and model version number. [Learn how to retrieve your project ID and model version number](https://docs.roboflow.com/api-reference/workspace-and-project-ids).
        2. `image_url`: The URL of the image you want to run inference on.
        3. `confidence`: The confidence threshold for predictions. Predictions with a confidence score below this threshold will be filtered out.
        4. `api_key`: Your Roboflow API key. [Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).
        5. `task`: The type of task you want to run. Choose from `object_detection`, `classification`, or `segmentation`.

        Then, run the Python script:

        ```
        python app.py
        ```

    === "NumPy Array"
    
        Create a new Python file and add the following code:

        ```python
        import requests
        import cv2
        import pickle

        project_id = "soccer-players-5fuqs"
        model_version = 1
        api_key = "YOUR API KEY"
        task = "object_detection"
        file_name = ""

        image = cv2.imread(file_name)
        numpy_data = pickle.dumps(image)

        res = requests.post(
            f"http://localhost:9001/{project_id}/{model_version}?api_key={api_key}&image_type=numpy",
            data=numpy_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        predictions = res.json()
        ```

        Above, specify:

        1. `project_id`, `model_version`: Your project ID and model version number. [Learn how to retrieve your project ID and model version number](https://docs.roboflow.com/api-reference/workspace-and-project-ids).
        2. `confidence`: The confidence threshold for predictions. Predictions with a confidence score below this threshold will be filtered out.
        3. `api_key`: Your Roboflow API key. [Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).
        4. `task`: The type of task you want to run. Choose from `object_detection`, `classification`, or `segmentation`.
        5. `filename`: The path to the image you want to run inference on.

        Then, run the Python script:

        ```
        python app.py
        ```

    === "Base64 Image"
    
        Create a new Python file and add the following code:

        ```python
        import requests
        import base64
        from PIL import Image

        project_id = ""
        model_version = ""
        image_url = ""
        confidence = 0.75
        api_key = ""
        task = "object_detection"
        file_name = ""

        image = Image.open(file_name)

        buffered = BytesIO()

        image.save(buffered, quality=100, format="JPEG")

        img_str = base64.b64encode(buffered.getvalue())

        infer_payload = {
            "model_id": f"{project_id}/{model_version}",
            "image": {
                "type": "base64",
                "value": img_str,
            },
            "confidence": confidence,
            "iou_threshold": iou_thresh,
            "api_key": api_key,
        }

        res = requests.post(
            f"http://localhost:9001/infer/{task}",
            json=infer_payload,
        )

        predictions = res.json()
        ```

        Above, specify:

        1. `project_id`, `model_version`: Your project ID and model version number. [Learn how to retrieve your project ID and model version number](https://docs.roboflow.com/api-reference/workspace-and-project-ids).
        2. `confidence`: The confidence threshold for predictions. Predictions with a confidence score below this threshold will be filtered out.
        3. `api_key`: Your Roboflow API key. [Learn how to retrieve your Roboflow API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key).
        4. `task`: The type of task you want to run. Choose from `object_detection`, `classification`, or `segmentation`.
        5. `filename`: The path to the image you want to run inference on.

        Then, run the Python script:

        ```
        python app.py
        ```

The code snippets above will run inference on a computer vision model. On the first request, the model weights will be downloaded and set up with your local inference server. This request may take some time depending on your network connection and the size of the model. Once your model has downloaded, subsequent requests will be much faster.

The Inference Server comes with a `/docs` route at `localhost:9001/docs` or `localhost:9001/redoc` that provides OpenAPI-powered documentation. You can use this to reference the routes available, and the configuration options for each route.

## Auto Batching Requests

Object detection models trained with Roboflow support batching, which allow you to upload multiple images of any type at once:

```python
infer_payload = {
    "model_id": f"{project_id}/{model_version}",
    "image": [
        {
            "type": "url",
            "value": image_url_1,
        },
        {
            "type": "url",
            "value": image_url_2,
        },
        {
            "type": "url",
            "value": image_url_3,
        },
    ],
    "confidence": confidence,
    "iou_threshold": iou_thresh,
    "api_key": api_key,
}
```