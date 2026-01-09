"""
Test script for Qwen3VL-2B-Instruct using inference-exp with PyTorch backend.

This script mirrors test_qwen3.py but uses the experimental inference library (inference-exp)
with explicit PyTorch backend for the vision-language model.
"""
import os

os.environ["DEVICE"] = "cuda"

from inference_exp import AutoModel
from PIL import Image

# Same API key and image as test_qwen3.py
API_KEY = "zaRavHwbvIXpGerDM3wi"
IMAGE = "dog-and-kitten.jpg"

# Load image
image = Image.open(IMAGE)

print("Testing Qwen3VL-2B-Instruct with Torch backend...")
print("=" * 60)

try:
    print("\nLoading qwen3vl-2b-instruct...")
    model = AutoModel.from_pretrained(
        "qwen3vl-2b-instruct",
        api_key=API_KEY,
    )

    # Run inference with a prompt
    prompt = "What's in this image?"
    print(f"Prompt: {prompt}")

    predictions = model.prompt(image, prompt=prompt)

    print(f"\nResponse: {predictions}")

except Exception as e:
    print(f"Error - {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Done!")
