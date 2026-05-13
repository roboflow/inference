<a href="https://www.yoloworld.cc/" target="_blank">YOLO-World</a> is a zero-shot object detection model.

You can use YOLO-World to identify objects in images and videos using arbitrary text prompts.

To use YOLO-World effectively, we recommend experimenting with the model to understand which text prompts help achieve the desired results.

YOLO World is faster than many other zero-shot object detection models like YOLO-World. On powerful hardware like a V100 GPU, YOLO World can run in real-time.

!!! note

    YOLO-World, like most state-of-the-art zero-shot detection models, is most effective at identifying common objects (i.e. cars, people, dogs, etc.). It is less effective at identifying uncommon objects (i.e. a specific type of car, a specific person, a specific dog, etc.).

!!! note

    In `inference` package YOLO-World models are identified by `yolo_world/<version>`, where `<version>` can be one of
    the following: `s`, `m`, `l`, `x`, `v2-s`, `v2-m`, `v2-l`, `v2-x`. Versions `v2-...` denote newer models, with
    improved evaluation metrics.


### How to Use YOLO-World

=== "Inference Python Library"

    Run the following command to set your API key in your coding environment:

    ```
    export ROBOFLOW_API_KEY=<your api key>
    ```

    Then, create a new Python file called `app.py` and add the following code:

    ```python
    import cv2
    import supervision as sv

    from inference.models.yolo_world.yolo_world import YOLOWorld

    image = cv2.imread("image.jpeg")

    model = YOLOWorld(model_id="yolo_world/l")
    classes = ["person", "backpack", "dog", "eye", "nose", "ear", "tongue"]
    results = model.infer("image.jpeg", text=classes, confidence=0.03)[0]

    detections = sv.Detections.from_inference(results)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [classes[class_id] for class_id in detections.class_id]

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    sv.plot_image(annotated_image)
    ```

=== "Inference Server HTTP API"

    Run the following command to set your API key in your coding environment:

    ```
    export ROBOFLOW_API_KEY=<your api key>
    ```

    Then, you will need to set up an Inference server to use the YOLO World HTTP API.

    To do this, run:

    ```
    pip install inference inference-sdk
    inference server start
    ```

    Then, create a new Python file called `app.py` and add the following code:

    ```python
    import os
    import cv2
    import supervision as sv

    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(
        api_url="http://127.0.0.1:9001",
        api_key=os.environ["ROBOFLOW_API_KEY"]
    )

    results = client.infer_from_yolo_world(
        inference_input=["https://media.roboflow.com/dog.jpeg"],
        class_names=["person", "backpack", "dog", "eye", "nose", "ear", "tongue"],
        model_version="l",
        confidence=0.1,
    )[0]

    detections = sv.Detections.from_inference(results)

    bounding_box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    labels = [classes[class_id] for class_id in detections.class_id]

    annotated_image = bounding_box_annotator.annotate(
        scene=image, detections=detections
    )
    annotated_image = label_annotator.annotate(
        scene=annotated_image, detections=detections, labels=labels
    )

    sv.plot_image(annotated_image)
    ```

=== "Inference Pipeline (Video)"

!!! Info
    
        **Breaking change!** There were versions: `0.9.14` and `0.9.15` where Yolo World was exposed
        behind `InferencePipeline.init(...)` initializer that you needed to run with specific combination 
        of parameters to alter default behavior of pipeline such that it runs against YoloWorld model. 
        We decided to provide an explicit way of running this foundation model in `InferencePipeline` providing
        a dedicated init function starting from version `0.9.16` 

    You can easily run predictions against `YoloWorld` model using `InferencePipeline`. There is a custom
    init method to ease handling that use-case:

    ```python
    # import the InferencePipeline interface
    from inference import InferencePipeline
    # import a built-in sink called render_boxes (sinks are the logic that happens after inference)
    from inference.core.interfaces.stream.sinks import render_boxes

    pipeline = InferencePipeline.init_with_yolo_world(
        video_reference="./your_video.mp4",
        classes=["person", "dog", "car", "truck"],
        model_size="s",
        on_prediction=render_boxes,
    )
    # start the pipeline
    pipeline.start()
    # wait for the pipeline to finish
    pipeline.join()
    ```

In this code, we load YOLO-World, run YOLO-World on an image, and annotate the image with the predictions from the model.

Above, replace:

1. `["person", "backpack", "dog", "eye", "nose", "ear", "tongue"]` with the objects you want to detect.
2. `image.jpeg` with the path to the image in which you want to detect objects.

Then, run the Python script you have created:

```
python app.py
```

The result from YOLO-World will be displayed in a new window.

![YOLO-World results](https://media.roboflow.com/yolo-world-dog.png)

### Benchmarking

We ran 100 inferences on an NVIDIA T4 GPU to benchmark the performance of YOLO-World.

YOLO-World ran 100 inferences in 9.18 seconds (0.09 seconds per inference, on average).