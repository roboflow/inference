from inference_sdk import InferenceHTTPClient
from PIL import Image

QWEN_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "InferenceImage", "name": "image"}],
    "steps": [
        {
            "type": "roboflow_core/qwen_vl@v1",
            "name": "model",
            "images": "$inputs.image",
            "prompt": "What is this food?",
            "model_version": "fooddescription/61",
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "model_output",
            "coordinates_system": "own",
            "selector": "$steps.model.raw_output",
        }
    ],
}

# Load image
image = Image.open("360_F_261982444_jDzDlgClqQDc5DX47Qy4PSayvcn89vQi.jpg")

# Initialize client with local VLM service URL
client = InferenceHTTPClient(
    api_url="http://localhost:9001",
    api_key="zaRavHwbvIXpGerDM3wi",
)

# Run workflow
result = client.run_workflow(
    specification=QWEN_WORKFLOW,
    images={"image": image},
)

print(result)
assert "model_output" in result[0]
assert isinstance(result[0]["model_output"], str)
assert len(result[0]["model_output"]) > 0