from inference import get_model
from PIL import Image
import os
os.environ["PROJECT"] = "roboflow-staging"
os.environ["ROBOFLOW_API_KEY"] = "GYaBMWQ6xDqVFsEJIoan"
os.environ["API_BASE_URL"] = "https://api.roboflow.one"

model = get_model("image-text/104", api_key="GYaBMWQ6xDqVFsEJIoan")
image = Image.open("test_img.jpg")
print(model.infer(image, prompt="What's in this image?"))