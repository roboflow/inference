[RF-DETR](https://blog.roboflow.com/rf-detr/) is a real-time object detection transformer-based architecture designed to transfer well to both a wide variety of domains and to datasets big and small.

RF-DETR is the first real-time model to exceed 60 AP on the Microsoft COCO benchmark alongside competitive performance at base sizes. It also achieves state-of-the-art performance on RF100-VL, an object detection benchmark that measures model domain adaptability to real world problems. RF-DETR is comparable speed to current real-time objection models.

![](https://media.roboflow.com/rf-detr/charts.png)

The model comes in two variants:

- RF-DETR Base, which has 29M parameters, and;
- RF-DETR Large, which has 129M parameters.

The RF-DETR source code and COCO checkpoint weights is available under an Apache 2.0 license.

## Supported Model Types

You can deploy the following RF-DETR model types with Inference:

- Object Detection

## Model Overview

- [RF-DETR background and architecture overview](https://blog.roboflow.com/rf-detr/)
- [Train an RF-DETR model on a custom dataset](https://blog.roboflow.com/train-rf-detr-on-a-custom-dataset/)
- [Train and deploy an RF-DETR model on Roboflow](https://blog.roboflow.com/train-deploy-rf-detr/)

## Usage Example

You can use RF-DETR with the following code:

```python
import os
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests

response = requests.get("https://media.roboflow.com/dog.jpeg")

if response.status_code == 200:
    image_data = BytesIO(response.content)

    image = Image.open(image_data)

model = get_model("rfdetr-base")

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

annotated_image.save("annotated_image_base.jpg")

model = get_model("rfdetr-large")

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

annotated_image.save("annotated_image_large.jpg")
```



If you have [fine-tuned an RF-DETR model on Roboflow](https://blog.roboflow.com/train-deploy-rf-detr/), you can deploy it by replacing the `rfdetr-base` model ID in the `get_model()` function call above with the ID of your trained model on Roboflow. This model ID will look like `model-name/1`, where `model-name` is the name of the model and `1` is your model version. [Learn how to find your model ID](https://docs.roboflow.com/api-reference/workspace-and-project-ids)

## License

See our [Licensing Guide](https://roboflow.com/licensing) for more information about how your use of RF-DETR is licensed when using Inference to deploy your model.
