
import requests
import base64
from PIL import Image
from io import BytesIO
import cv2
import pickle

project_id = "soccer-players-5fuqs"
model_version = 1
confidence = 0.5
iou_thresh = 0.5
api_key = "dH5yk9Xs7XmeCa1Mv7XX"
file_name = "people-walking.jpg"

image = cv2.imread(file_name)
numpy_data = pickle.dumps(image)

res = requests.post(
f"http://localhost:9001/{project_id}/{model_version}?api_key={api_key}&image_type=numpy",
data=numpy_data,
headers={"Content-Type": "application/x-www-form-urlencoded"},
)

predictions = res.json()
print(predictions)