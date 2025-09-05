import os
import io
import cv2
import numpy as np
import requests
from PIL import Image
import supervision as sv
from inference import get_model

os.environ["ONNXRUNTIME_EXECUTION_PROVIDERS"] = "['TensorrtExecutionProvider']"

print("getting model")
model = get_model("rfdetr-nano")

IMAGE_URL = "https://media.roboflow.com/notebooks/examples/dog-2.jpeg"
image_pil = Image.open(io.BytesIO(requests.get(IMAGE_URL).content)).convert("RGB")
image_np = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)

print("Got model, running inference")
pred = model.infer(image_pil, confidence=0.3)[0]
print(pred)

detections = sv.Detections.from_inference(pred)
labels = [p.class_name for p in pred.predictions]

box_annotator = sv.BoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)

annotated = box_annotator.annotate(image_np.copy(), detections)
annotated = label_annotator.annotate(annotated, detections, labels=labels)

cv2.imwrite("annotated_image.jpg", annotated)