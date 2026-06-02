---
description: Find and run community-trained models from Roboflow Universe - 50,000+ public computer vision models published by other Roboflow users.
---

# Universe Models

[Roboflow Universe](https://universe.roboflow.com) is a public catalog of **50,000+ computer vision models published by other Roboflow users** - covering everything from defect detection and sports analytics to wildlife identification. This page shows how to pick one of those community models and run it with Inference.

## Run a Model From Roboflow Universe

Let's pick a community model from Universe and run it.

Go to the <a href="https://universe.roboflow.com" target="_blank">Roboflow Universe</a> homepage and use the search bar to find a model.

![Roboflow Universe search bar](https://media.roboflow.com/universe-search.png)

!!! info

    Add "model" to your search query to only find models.

Browse the search page to find a model.

![Search page](https://media.roboflow.com/universe-search-page.png)

When you have found a model, click on the model card to learn more. Click the "Model" link in the sidebar to get the information you need to use the model.

Next, create a new Python file and add the following code. See [Run a Model](./run_a_model.md) for package installation (including GPU and HTTP-client variants) and [Supervision](https://supervision.roboflow.com) for the visualization helpers used below.

=== "inference-sdk (HTTP client)"

    ```python
    # import the HTTP client for sending inference requests to an Inference Server
    from inference_sdk import InferenceHTTPClient
    # import supervision to visualize our results
    import supervision as sv
    # import cv2 to help load our image
    import cv2

    # define the image url to use for inference
    image_file = "people-walking.jpg"
    image = cv2.imread(image_file)

    # connect to an Inference Server (Roboflow-hosted or self-hosted)
    client = InferenceHTTPClient(
        # api_url="http://localhost:9001",  # for Self-hosted
        api_url="https://serverless.roboflow.com",
        api_key="ROBOFLOW_API_KEY",
    )

    # Run the inference with cup-detection-cevbw/1 (Yolov8s) model from Universe
    results = client.infer(image, model_id="cup-detection-cevbw/1")

    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results)

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # display the image
    sv.plot_image(annotated_image)
    ```

=== "inference (native)"

    ```python
    # import a utility function for loading Roboflow models
    from inference import get_model
    # import supervision to visualize our results
    import supervision as sv
    # import cv2 to help load our image
    import cv2

    # define the image url to use for inference
    image_file = "people-walking.jpg"
    image = cv2.imread(image_file)

    # load the cup-detection-cevbw/1 (Yolov8s) from Universe
    model = get_model(model_id="cup-detection-cevbw/1")

    # run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
    results = model.infer(image)

    # load the results into the supervision Detections api
    detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

    # create supervision annotators
    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    # annotate the image with our inference results
    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections)
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections)

    # display the image
    sv.plot_image(annotated_image)
    ```


!!! Tip

    To see more models, check out the [Pre-Trained Models](../quickstart/aliases.md) page and [Roboflow Universe](https://universe.roboflow.com).

The `people-walking.jpg` file is hosted <a href="https://media.roboflow.com/inference/people-walking.jpg" target="_blank">here</a>.

Replace `rfdetr-small` with the model ID you found on Universe, replace `image` with the image of your choosing, and be sure to export your API key:

```
export ROBOFLOW_API_KEY=<your api key>
```

Then, run the Python script:

```
python app.py
```

You should see your model's predictions visualized on your screen.

![People Walking Annotated](https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking-annotated.jpg)
