With Inference, you can run any of the 50,000+ models available on Roboflow Universe.

All models run on your own hardware.

## Run a Model From Roboflow Universe

In the first example, we showed how to run a people detection model. This model was hosted on Universe. Let's find another model to try.

Go to the <a href="https://universe.roboflow.com" target="_blank">Roboflow Universe</a> homepage and use the search bar to find a model.

![Roboflow Universe search bar](https://media.roboflow.com/universe-search.png)

!!! info

    Add "model" to your search query to only find models.

Browse the search page to find a model.

![Search page](https://media.roboflow.com/universe-search-page.png)

When you have found a model, click on the model card to learn more. Click the "Model" link in the sidebar to get the information you need to use the model.

Then, install Inference and supervision, which we will use to run our model and handle model predictions, respectively:

```bash
pip install inference supervision
```

Next, create a new Python file and add the following code:

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

    To see more models, check out the [Pre-Trained Models](/quickstart/aliases) page and [Roboflow Universe](https://universe.roboflow.com).

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
