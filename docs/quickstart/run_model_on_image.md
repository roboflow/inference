A Roboflow Inference server provides a standard API through which to run inference on computer vision models.

In this guide, we show how to run inference on object detection, classification, and segmentation models using Inference.

!!! note

    Inference is compatible with models trained on Roboflow, but stay tuned as we actively develop support for bringing your own models.

You can run inference on images from:

1. URLs, which will be downloaded from the internet
2. File names, which will be read from disk
3. PIL images
4. NumPy arrays

## Step #1: Install the Inference Server

_You can skip this step if you already have Inference installed and running._

The Inference Server runs in Docker. Before we begin, make sure you have installed Docker on your system. To learn how to install Docker, refer to the [official Docker installation guide](https://docs.docker.com/get-docker/).

Once you have Docker installed, you are ready to download Roboflow Inference. The command you need to run depends on what device you are using.

Start the server using `inference server start`. After you have installed the Inference Server, the Docker container will start running the server at `localhost:9001`.

## Step #2: Run Inference

You can send a URL with an image, a NumPy array, or a base64-encoded image to an Inference server. The server will return a JSON response with the predictions.

There are two generations of routes in a Roboflow inference server To see what routes are available for a running inference server instance, visit the `/docs` route in a browser. Roboflow hosted inference endpoints (`detect.roboflow.com`) only support V1 routes.

### Run Inference on a v2 Route

!!! Run Inference

    === "URL"

        Create a new Python file and add the following code:

        ```python
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration

        project_id = "soccer-players-5fuqs"
        model_version = 1
        image_url = "https://media.roboflow.com/inference/soccer.jpg"
        confidence = 0.75

        custom_configuration = InferenceConfiguration(confidence_threshold=confidence)

        client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=os.environ["API_KEY"],
        )

        client.select_api(model_version)

        results = client.infer(image_url, model_id=f"{project_id}/{model_version}")

        print(predictions)
        ```

        Above, specify:

        1. `project_id`, `model_version`: Your project ID and model version number. [Learn how to retrieve your project ID and model version number](https://docs.roboflow.com/api-reference/workspace-and-project-ids).
        2. `image_url`: The URL of the image you want to run inference on.
        3. `confidence`: The confidence threshold for predictions. Predictions with a confidence score below this threshold will be filtered out.

        Then, run the Python script:

        ```
        python app.py
        ```

    === "PIL Image"

        Create a new Python file and add the following code:

        ```python
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        from PIL import Image

        project_id = "soccer-players-5fuqs"
        model_version = 1
        filename = "path/to/local/image.jpg"
        confidence = 0.75

        custom_configuration = InferenceConfiguration(confidence_threshold=confidence)

        client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=os.environ["API_KEY"],
        )

        client.select_api(model_version)

        results = client.infer(Image.open(filename), model_id=f"{project_id}/{model_version}")

        print(predictions)
        ```

        Above, specify:

        1. `project_id`, `model_version`: Your project ID and model version number. [Learn how to retrieve your project ID and model version number](https://docs.roboflow.com/api-reference/workspace-and-project-ids).
        2. `confidence`: The confidence threshold for predictions. Predictions with a confidence score below this threshold will be filtered out.
        3. `filename`: The path to the image you want to run inference on.

        Then, run the Python script:

        ```
        python app.py
        ```

    === "NumPy Array"

        Create a new Python file and add the following code:

        ```python
        from inference_sdk import InferenceHTTPClient, InferenceConfiguration
        import cv2

        project_id = "soccer-players-5fuqs"
        model_version = 1
        filename = "path/to/local/image.jpg"
        confidence = 0.75

        custom_configuration = InferenceConfiguration(confidence_threshold=confidence)

        client = InferenceHTTPClient(
            api_url="http://localhost:9001",
            api_key=os.environ["API_KEY"],
        )

        client.select_api(model_version)

        results = client.infer(cv2.imread(filename), model_id=f"{project_id}/{model_version}")

        print(predictions)
        ```

        Above, specify:

        1. `project_id`, `model_version`: Your project ID and model version number. [Learn how to retrieve your project ID and model version number](https://docs.roboflow.com/api-reference/workspace-and-project-ids).
        2. `confidence`: The confidence threshold for predictions. Predictions with a confidence score below this threshold will be filtered out.
        3. `filename`: The path to the image you want to run inference on.

        Then, run the Python script:

        ```
        python app.py
        ```


    === "Batch Inference"

        Roboflow object detection models support batching. Utilize batch inference by passing a list of image objects in a request payload:

        ```python
        import requests
        import base64
        from PIL import Image
        from io import BytesIO

        project_id = "soccer-players-5fuqs"
        model_version = 1
        task = "object_detection"
        confidence = 0.5
        iou_thresh = 0.5
        api_key = "YOUR ROBOFLOW API KEY"
        file_name = "path/to/local/image.jpg"

        image = Image.open(file_name)

        buffered = BytesIO()

        image.save(buffered, quality=100, format="JPEG")

        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        infer_payload = {
            "model_id": f"{project_id}/{model_version}",
            "image": [
                {
                    "type": "base64",
                    "value": img_str,
                },
                {
                    "type": "base64",
                    "value": img_str,
                },
                {
                    "type": "base64",
                    "value": img_str,
                }
            ],
            "confidence": confidence,
            "iou_threshold": iou_thresh,
            "api_key": api_key,
        }

        res = requests.post(
            f"http://localhost:9001/infer/{task}",
            json=infer_payload,
        )

        predictions = res.json()
        print(predictions)
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

### Run Inference on a v1 Route

!!! Run Inference
=== "URL"

        The Roboflow hosted API uses the V1 route and requests take a slightly different form:

        ```python
        import requests
        import base64
        from PIL import Image
        from io import BytesIO

        project_id = "soccer-players-5fuqs"
        model_version = 1
        confidence = 0.5
        iou_thresh = 0.5
        api_key = "YOUR ROBOFLOW API KEY"
        image_url = "https://storage.googleapis.com/com-roboflow-marketing/inference/soccer.jpg


        res = requests.post(
            f"https://detect.roboflow.com/{project_id}/{model_version}?api_key={api_key}&confidence={confidence}&overlap={iou_thresh}&image={image_url}",
        )

        predictions = res.json()
        print(predictions)
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

        The Roboflow hosted API uses the V1 route and requests take a slightly different form:

        ```python
        import requests
        import base64
        from PIL import Image
        from io import BytesIO

        project_id = "soccer-players-5fuqs"
        model_version = 1
        confidence = 0.5
        iou_thresh = 0.5
        api_key = "YOUR ROBOFLOW API KEY"
        file_name = "path/to/local/image.jpg"

        image = Image.open(file_name)

        buffered = BytesIO()

        image.save(buffered, quality=100, format="JPEG")

        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")

        res = requests.post(
            f"https://detect.roboflow.com/{project_id}/{model_version}?api_key={api_key}&confidence={confidence}&overlap={iou_thresh}",
            data=img_str,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        predictions = res.json()
        print(predictions)
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

    === "NumPy Array"

        Numpy arrays can be pickled and passed to the inference server for quicker processing. Note, Roboflow hosted APIs to not accept numpy inputs for security reasons:

        ```python
        import requests
        import cv2
        import pickle

        project_id = "soccer-players-5fuqs"
        model_version = 1
        task = "object_detection"
        api_key = "YOUR API KEY"
        file_name = "path/to/local/image.jpg"

        image = cv2.imread(file_name)
        numpy_data = pickle.dumps(image)

        res = requests.post(
            f"http://localhost:9001/{project_id}/{model_version}?api_key={api_key}&image_type=numpy",
            data=numpy_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )

        predictions = res.json()
        print(predictions)
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

    === "Batch Inference"

       Batch inference is not currently supported by V1 routes.

The code snippets above will run inference on a computer vision model. On the first request, the model weights will be downloaded and set up with your local inference server. This request may take some time depending on your network connection and the size of the model. Once your model has downloaded, subsequent requests will be much faster.

The Inference Server comes with a `/docs` route at `localhost:9001/docs` or `localhost:9001/redoc` that provides OpenAPI-powered documentation. You can use this to reference the routes available, and the configuration options for each route.

## Batching Requests

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
