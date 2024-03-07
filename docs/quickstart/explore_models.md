With Inference, you can run any of the 50,000+ models available on Roboflow Universe. You can also run private, fine-tuned models that you have trained or uploaded to Roboflow.

All models run on your own hardware.

## Run pre-trained YOLOv8 model

Roboflow Universe exposes pre-trained YOLOv8 models.

- object-detection: `yolov8{model_size}-{inference_resolution}`

  - supported model sizes: `[n, s, m, l, x]`
  - supported inference resolutions: `[640, 1280]`

- instance-segmentation: `yolov8{model_size}-seg-{inference_resolution}`
  - supported model sizes: `[n, s, m, l, x]`
  - supported inference resolutions: `[640, 1280]`

## Run a Model on Universe

In the first example, we showed how to run a rock paper scissors model. This model was hosted on Universe. Let's find another model to try.

!!! Info

    If you haven't already, follow our [Run Your First Model guide](http://127.0.0.1:8000/quickstart/run_a_model/) to install and set up Inference.

Go to the <a href="https://universe.roboflow.com" target="_blank">Roboflow Universe</a> homepage and use the search bar to find a model.

![Roboflow Universe search bar](https://media.roboflow.com/universe-search.png)

!!! info

    Add "model" to your search query to only find models.

Browse the search page to find a model.

![Search page](https://media.roboflow.com/universe-search-page.png)

When you have found a model, click on the model card to learn more. Click the "Model" link in the sidebar to get the information you need to use the model.

Create a new Python file and add the following code:

```python
# import a utility function for loading Roboflow models
from inference import get_model
# import supervision to visualize our results
import supervision as sv
# import cv2 to helo load our image
import cv2

# define the image url to use for inference
image_file = "people-walking.jpg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_model(model_id="yolov8n-640")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)
```

The `people-walking.jpg` file is hosted <a href="https://media.roboflow.com/inference/people-walking.jpg" target="_blank">here</a>.

Replace `yolov8n-640` with the model ID you found on Universe, replace `image` with the image of your choosing, and be sure to export your API key:

```
export ROBOFLOW_API_KEY=<your api key>
```

Then, run the Python script:

```
python app.py
```

You should see your model's predictions visualized on your screen.

![People Walking Annotated](https://storage.googleapis.com/com-roboflow-marketing/inference/people-walking-annotated.jpg)

## Run a Private, Fine-Tuned Model

You can run models you have trained privately on Roboflow with Inference. To do so, first go to your <a href="https://app.roboflow.com" target="_blank">Roboflow dashboard</a>. Then, choose the model you want to run.

![Roboflow dashboard](https://media.roboflow.com/docs-models.png)

Click the "Deploy" link in the sidebar to find the information you will need to use your model with Inference.

Copy the model ID on the page (in this case, `taylor-swift-records/3`).

![Model page](https://media.roboflow.com/docs-model-id.png)

Then, create a new Python file and add the following code:

```python
# import a utility function for loading Roboflow models
from inference import get_model
# import supervision to visualize our results
import supervision as sv
# import cv2 to helo load our image
import cv2

# define the image url to use for inference
image_file = "taylor-swift-album-1989.jpeg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_model(model_id="taylor-swift-records/3")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)

# load the results into the supervision Detections api
detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

# create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# annotate the image with our inference results
annotated_image = bounding_box_annotator.annotate(
    scene=image, detections=detections)
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections)

# display the image
sv.plot_image(annotated_image)
```

The `taylor-swift-album-1989.jpeg` file is hosted <a href="https://storage.googleapis.com/com-roboflow-marketing/inference/taylor-swift-album-1989.jpeg" target="_blank">here</a>.

Replace `taylor-swift-records/3` with the model ID from your private model and ensure your API key is in your environment:

```
export ROBOFLOW_API_KEY=<your api key>
```

Then, run the Python script:

```
python app.py
```

You should see your model's predictions visualized on your screen.

![Taylor Swift Album](https://storage.googleapis.com/com-roboflow-marketing/inference/taylor-swift-album-1989-annotated.jpeg)
