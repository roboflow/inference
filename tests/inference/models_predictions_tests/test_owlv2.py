from inference.core.entities.requests.owlv2 import OwlV2InferenceRequest
from inference.core.entities.responses.inference import ObjectDetectionInferenceResponse
from inference.models.owlv2.owlv2 import OwlV2

image={"type": "url", "value": "https://media.roboflow.com/inference/seawithdock.jpeg"}
request = OwlV2InferenceRequest(
    image=image,
    training_data=[
        {"image": image,
         "boxes": [
             {"x": 223,
              "y": 306,
              "w": 40,
              "h": 226,
              "cls": "post"}
         ]}
    ],
    visualize_predictions=True
)

import requests
# response = OwlV2().infer_from_request(request)
response = requests.post("http://localhost:9001/owlv2/infer", json=request.dict())
response = ObjectDetectionInferenceResponse(**response.json()) 
from PIL import Image
import io
import base64

def load_image_from_base64(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data))
    return image

print(type(response.visualization))
visualization = load_image_from_base64(response.visualization)
visualization.save("owlvit.jpg")