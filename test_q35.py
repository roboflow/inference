
import os
#os.environ["PROJECT"] = "roboflow-staging"
#os.environ["API_BASE_URL"] = "https://api.roboflow.one"
os.environ["ROBOFLOW_API_KEY"] = "zaRavHwbvIXpGerDM3wi"
#os.environ["ROBOFLOW_ENVIRONMENT"] = "staging" 
os.environ["ROBOFLOW_API_KEY"] = "zaRavHwbvIXpGerDM3wi"

from inference_models import AutoModel
from PIL import Image

# Load the model
model = AutoModel.from_pretrained(
    "qwen3_5-0.8b",
    api_key="zaRavHwbvIXpGerDM3wi",
    verbose=True,  # Enable verbose to see what's happening
)

# Load a test image
image = Image.open("/home/matveipopov/inference/IMG_2346.jpeg")  # Replace with actual image path
print("Image loaded successfully.")
# Run inference
result = model.prompt(
    images=image,
    prompt="Is there any animals in the image?",
)

print(result)