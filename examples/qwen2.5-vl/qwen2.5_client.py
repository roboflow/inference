import base64
import requests
import os

PORT = 9001
IMAGE_PATH = "../cogvlm/image.jpg"

def encode_base64(image_path):
    with open(image_path, "rb") as image:
        x = image.read()
        image_string = base64.b64encode(x)

    return image_string.decode("ascii")

def do_qwen_request():
    # Basic caption prompt
    prompt = "Caption this image in detail"
    
    # For object detection, use:
    # prompt = "Detect all objects in this image and return their locations in JSON format"
    
    # For landmark recognition, use:
    # prompt = "What landmarks can you identify in this image? Give their names in both English and Chinese."
    
    # For OCR, use:
    # prompt = "Extract all text from this image"
    
    # You can also add a system prompt (optional)
    # system_prompt = "You are an expert in visual analysis"
    # combined_prompt = f"{prompt}<system_prompt>{system_prompt}"
    # prompt = combined_prompt  # If using system prompt, uncomment this line
    
    # Note: If no system prompt is provided, the model will use its default:
    # "You are a Qwen2.5-VL model that can answer questions about any image."

    print("Starting request to Qwen2.5 VL...")
    infer_payload = {
        "image": {
            "type": "base64",
            "value": encode_base64(IMAGE_PATH),
        },
        "prompt": prompt,
        "model_id": "qwen2.5-vl-7b",  # Can also use qwen2.5-vl-3b or qwen2.5-vl-72b
    }
    
    response = requests.post(
        f'http://localhost:{PORT}/infer/lmm',
        json=infer_payload,
    )
    resp = response.json()
    print(resp)


if __name__ == "__main__":
    do_qwen_request()