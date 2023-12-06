You can run fine-tuned models on images using Inference.

An Inference server will manage inference. Inference can be run on your local machine, a remote server, or even a Raspberry Pi.

If you need to deploy to the edge, you can use a device like the Jetson Nano. If you need high-performance compute for batch jobs, you can deploy Inference to a server with a GPU.

!!! tip "Tip"
    Follow our [Run a Fine-Tuned Model on Images](/docs/quickstart/run_model_on_image) guide to learn how to find a model to run.

!!! info
    If you haven't already, follow our Run Your First Model guide to install and set up Inference.

Create a new Python file and add the following code:

```python
from inference.models.utils import get_roboflow_model
from PIL import Image

model = get_roboflow_model(
    model_id="soccer-players-5fuqs/1",
    api_key="YOUR ROBOFLOW API KEY"
)

result = model.infer("path/to/image.jpg")

print(result)
```

Replace your API key, model ID, and model version as appropriate.

- [Learn how to find your API key](https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key)
- [Learn how to find your model ID](https://docs.roboflow.com/api-reference/workspace-and-project-ids)

Then, run the code. You will see predictions printed to the console in the following format:

```json
{
    "predictions": [
        {
            "class": "rock",
            "confidence": 0.9999997615814209,
            "height": 0.9999997615814209,
            "width": 0.9999997615814209,
            "x": 0.0,
            "y": 0.0
        }
    ]
}
```

You can plot predictions using `supervision`. You can install supervision using `pip install supervision`. Add the following code to the script you created to plot predictions from Inference:

```python
import supervision as sv

detections = sv.Detections.from_roboflow(results)
labels = [p["class"] for p in predictions["predictions"]]

box_annotator = sv.BoxAnnotator()

annotated_frame = box_annotator.annotate(
	scene=image.copy(),
	detections=detections,
    labels=labels
)

sv.plot_image(image=annotated_frame, size=(16, 16))
```