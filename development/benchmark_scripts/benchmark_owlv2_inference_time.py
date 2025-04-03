import tempfile
import time

import requests
from PIL import Image

from inference.models.owlv2.owlv2 import OwlV2

# run a simple latency test
image_via_url = {
    "type": "url",
    "value": "https://media.roboflow.com/inference/seawithdock.jpeg",
}

# Download the image
response = requests.get(image_via_url["value"])
response.raise_for_status()

# Create a temporary file
with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
    temp_file.write(response.content)
    temp_file_path = temp_file.name

img = Image.open(temp_file_path)
img = img.convert("RGB")

request_dict = dict(
    image=img,
    training_data=[
        {
            "image": img,
            "boxes": [{"x": 223, "y": 306, "w": 40, "h": 226, "cls": "post", "negative": False}],
        }
    ],
    visualize_predictions=False,
)

model = OwlV2()

for _ in range(10):
    print("pre cache fill try")
    time_start = time.time()
    response = model.infer(**request_dict)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

    print("post cache fill try")
    time_start = time.time()
    response = model.infer(**request_dict)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")

    model.reset_cache()