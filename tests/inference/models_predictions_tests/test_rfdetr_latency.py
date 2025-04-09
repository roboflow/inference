import time
from io import BytesIO

import requests
from PIL import Image

from inference import get_model

response = requests.get("https://media.roboflow.com/dog.jpeg")

if response.status_code == 200:
    image_data = BytesIO(response.content)
    image = Image.open(image_data)

model = get_model("rfdetr-base")

times = []
for i in range(100):
    start_time = time.time()
    model.infer(image, confidence=0.5)
    end_time = time.time()

    elapsed_time = end_time - start_time

    if i > 50:
        times.append(elapsed_time)
    print(f"Run {i+1}: {elapsed_time:.4f} seconds")

average_time = sum(times) / len(times)

print(f"\nAverage inference time Base: {average_time:.4f} seconds")

model = get_model("rfdetr-large")

times = []
for i in range(100):
    start_time = time.time()
    model.infer(image, confidence=0.5)
    end_time = time.time()

    elapsed_time = end_time - start_time

    if i > 50:
        times.append(elapsed_time)
    print(f"Run {i+1}: {elapsed_time:.4f} seconds")

average_time = sum(times) / len(times)

print(f"\nAverage inference time Large: {average_time:.4f} seconds")
