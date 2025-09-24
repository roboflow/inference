import base64
import os

import requests

PORT = 9001
IMAGE_PATH = "../cogvlm/image.jpg"

def encode_bas64(image_path):
    with open(image_path, "rb") as image:
        x = image.read()
        image_string = base64.b64encode(x)

    return image_string.decode("ascii")

def do_gemma_request():
    prompt = "Caption"

    print(f"Starting")
    infer_payload = {
        "image": {
            "type": "base64",
            "value": encode_bas64(IMAGE_PATH),
        },
        "prompt": prompt,
        "model_id": "paligemma-3b-mix-224",
    }
    response = requests.post(
        f'http://localhost:{PORT}/infer/lmm',
        json=infer_payload,
    )
    resp = response.json()
    print(resp)


if __name__ == "__main__":
    do_gemma_request()