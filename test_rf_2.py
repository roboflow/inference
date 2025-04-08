import os
import supervision as sv
from inference import get_model
from PIL import Image
import io
import requests
import time
import numpy as np

IMAGE_PATH = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
model = get_model("rfdetr-base")

all_times = []
warmup_count = 50

for i in range(100):
    if i < warmup_count:
        image = Image.open(io.BytesIO(requests.get(IMAGE_PATH).content))
        predictions = model.infer(image, confidence=0.5)[0]
    else:
        start_time = time.perf_counter()
        image = Image.open(io.BytesIO(requests.get(IMAGE_PATH).content))
        predictions = model.infer(image, confidence=0.5)[0]
        end_time = time.perf_counter()
        all_times.append(end_time - start_time)

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

print(f"Average time taken: {np.mean(all_times)} seconds")
sv.plot_image(annotated_image)

annotated_image.save("annotated_image_base_cut.jpg")

"""
model = get_model("rfdetr-large")

predictions = model.infer(image, confidence=0.5)[0]

detections = sv.Detections.from_inference(predictions)

labels = [prediction.class_name for prediction in predictions.predictions]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator().annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

sv.plot_image(annotated_image)

annotated_image.save("annotated_image_large.jpg")
"""