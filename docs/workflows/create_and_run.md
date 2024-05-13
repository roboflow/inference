Workflows allow you to define multi-step processes that run one or more models and return a result based on the output of the models.


You can create and deploy workflows in the cloud or in Inference.

To create an advanced workflow for use with Inference, you need to define a specification. A specification is a JSON document that states:

1. The version of workflows you are using.
2. The expected inputs.
3. The steps the workflow should run (i.e. models to run, filters to apply, etc.).
4. The expected output format.

In this guide, we walk through how to create a basic workflow that completes three steps:

1. Run a model to detect objects in an image.
2. Crops each region.
3. Runs OCR on each region.

You can use the guidance below as a template to learn the structure of workflows, or verbatim to create your own detect-then-OCR workflows.

## Step #1: Define an Input

Workflows require a specification to run. A specification takes the follwoing form:

```python
SPECIFICATION = {
    "specification": {
        "version": "1.0",
        "inputs": [],
        "steps": [],
        "outputs": []
    }
}
```

Within this structure, we need to define our:

1. Model inputs
2. The steps to run
3. The expected output

First, let's define our inputs.

For this workflow, we will specify an image input:

```json
"steps": [
    { "type": "InferenceImage", "name": "image" },   # definition of input image
]
```

## Step #2: Define Processing Steps

Next, we need to define our processing steps. For this guide, we want to:

1. Run a model to detect license plates.
2. Crop each license plate.
3. Run OCR on each license plate.

We can define these steps as follows:

```json
"steps": [
    {
        "type": "ObjectDetectionModel",   # definition of object detection model
        "name": "plates_detector",  
        "image": "$inputs.image",  # linking input image into detection model
        "model_id": "vehicle-registration-plates-trudk/2",  # pointing model to be used
    },
        {
        "type": "DetectionOffset",  # DocTR model usually works better if there is slight padding around text to be detected - hence we are offseting predictions
        "name": "offset",
        "predictions": "$steps.plates_detector.predictions",  # reference to the object detection model output
        "offset_x": 200,  # value of offset
        "offset_y": 40,  # value of offset
    },
    {
        "type": "Crop",   # we would like to run OCR against each and every plate detected - hece we are cropping inputr image using offseted predictions
        "name": "cropping",
        "image": "$inputs.image",  # we need to point image to crop
        "detections": "$steps.offset.predictions",  # we need to point detections that will be used to crop image (in this case - we use offseted prediction)
    },        
    {
        "type": "OCRModel",  # we define OCR model
        "name": "step_ocr",
        "image": "$steps.cropping.crops",  # OCR model as an input takes a reference to crops that were created based on detections
    },
],
```

## Step #3: Define an Output

Finally, we need to define the output for our workflow:

```json
"outputs": [
    { "type": "JsonField", "name": "predictions", "selector": "$steps.plates_detector.predictions" },  # output with object detection model predictions
    { "type": "JsonField", "name": "image", "selector": "$steps.plates_detector.image" },  # output with image metadata - required by `supervision`
    { "type": "JsonField", "name": "recognised_plates", "selector": "$steps.step_ocr.result" },  # field that will retrieve OCR result
    { "type": "JsonField", "name": "crops", "selector": "$steps.cropping.crops" },  # crops that were made based on plates detections - used here just to ease visualisation
]   
```

## Step #4: Run Your Workflow

Now that we have our specification, we can run our workflow using the Inference SDK.

=== "Run Locally with Inference"

    Use `inference_cli` to start server

    ```bash
    inference server start
    ```

    ```python
    from inference_sdk import InferenceHTTPClient, VisualisationResponseFormat, InferenceConfiguration
    import supervision as sv
    import cv2
    from matplotlib import pyplot as plt

    client = InferenceHTTPClient(
        api_url="http://127.0.0.1:9001",
        api_key="YOUR_API_KEY"
    )

    client.configure(
        InferenceConfiguration(output_visualisation_format=VisualisationResponseFormat.NUMPY)
    )

    license_plate_image_1 = cv2.imread("./images/license_plate_1.jpg")

    license_plate_result_1 = client.infer_from_workflow(
        specification=READING_PLATES_SPECIFICATION["specification"],
        images={"image": license_plate_image_1},
    )

    plt.title(f"Recognised plate: {license_plate_result_1['recognised_plates']}")
    plt.imshow(license_plate_result_1["crops"][0]["value"][:, :, ::-1])
    plt.show()
    ```

    Here are the results:

    ![Recognised plate: "34 6511"](https://media.roboflow.com/inference/license_plate_1.png)

=== "Run in the Roboflow Cloud"

    ```python
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="https://detect.roboflow.com",
        api_key="YOUR_API_KEY"
    )

    client.configure(
        InferenceConfiguration(output_visualisation_format=VisualisationResponseFormat.NUMPY)
    )

    license_plate_image_1 = cv2.imread("./images/license_plate_1.jpg")

    license_plate_result_1 = client.infer_from_workflow(
        specification=READING_PLATES_SPECIFICATION["specification"],
        images={"image": license_plate_image_1},
    )

    plt.title(f"Recognised plate: {license_plate_result_1['recognised_plates']}")
    plt.imshow(license_plate_result_1["crops"][0]["value"][:, :, ::-1])
    plt.show()
    ```

    Here are the results:

    ![Recognised plate: "34 6511"](https://media.roboflow.com/inference/license_plate_1.png)

## Next Steps

Now that you have created and run your first workflow, you can explore our other supported blocks and create a more complex workflow.

Refer to our [Supported Blocks](/workflows/supported_blocks/) documentation to learn more about what blocks are supported.