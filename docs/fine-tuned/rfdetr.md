RF-DETR is Roboflow's new Real-Time SOTA Object Detector.

## Supported Model Types

You can deploy the following RF-DETR model types with Inference:

- Object Detection

## Model Overview

- [Read More About RF-DETR](https://blog.roboflow.com/rf-detr/)

## Usage Example

```
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

## License

See our [Licensing Guide](https://roboflow.com/licensing) for more information about how your use of RF-DETR is licensed when using Inference to deploy your model.