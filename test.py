import os
import supervision as sv
from inference import get_model
from inference.models.rfdetr.coco_classes import COCO_CLASSES
from PIL import Image
image = Image.open("demo.jpg")

model = get_model("rfdetr-base", api_key="CV6vJpMITzC6LraZfqQr")

predictions = model.infer(image)[0]

detections = sv.Detections.from_inference(predictions)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

annotated_image.save("annotated_image_base.jpg")

model = get_model("rfdetr-large", api_key="CV6vJpMITzC6LraZfqQr")

predictions = model.infer(image)[0]

detections = sv.Detections.from_inference(predictions)

labels = [
    f"{COCO_CLASSES[class_id]} {confidence:.2f}"
    for class_id, confidence
    in zip(detections.class_id, detections.confidence)
]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

annotated_image.save("annotated_image_large.jpg")