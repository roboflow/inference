With Inference, you can run private, fine-tuned models that you have trained or uploaded to Roboflow.

All models run on your own hardware.

## Run a Private, Fine-Tuned Model

To run a model, first go to your <a href="https://app.roboflow.com" target="_blank">Roboflow dashboard</a>. Then, choose the model you want to run.

![Roboflow dashboard](https://media.roboflow.com/docs-models.png)

Click the "Deploy" link in the sidebar to find the information you will need to use your model with Inference.

Copy the model ID on the page (in this case, `taylor-swift-records/3`).

![Model page](https://media.roboflow.com/docs-model-id.png)

Then, create a new Python file and add the following code:

```python
from inference import get_model
import supervision as sv
import cv2

# define the image url to use for inference
image_file = "taylor-swift-album-1989.jpeg"
image = cv2.imread(image_file)

# load a pre-trained yolov8n model
model = get_model(model_id="taylor-swift-records/3")

# run inference on our chosen image, image can be a url, a numpy array, a PIL image, etc.
results = model.infer(image)[0]

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
