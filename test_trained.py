import os
import supervision as sv
from inference import get_model
from PIL import Image
from io import BytesIO
import requests

image_data = "airport_39_jpg.rf.50aac3cc9c5d94f56a580eeed8cdd02c.jpg"

image = Image.open(image_data)

model = get_model("aerial-airport-7ap9o-ddgc-ftba6-eslve/3", api_key="zaRavHwbvIXpGerDM3wi")

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

annotated_image.save("annotated_image_base.jpg")