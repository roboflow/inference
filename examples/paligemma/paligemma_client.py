import base64
import requests
import os

PORT = 9001
API_KEY = os.environ["API_KEY"]
IMAGE_PATH = "../cogvlm/image.jpg"

def encode_bas64(image_path):
    with open(image_path, "rb") as image:
        x = image.read()
        image_string = base64.b64encode(x)

    return image_string.decode("ascii")

def do_gemma_request():
    prompt = "Describe this image"

    print(f"Starting")
    infer_payload = {
        "image": {
            "type": "base64",
            "value": encode_bas64("youtube-19-small.jpg"),
        },
        "api_key": API_KEY,
        "prompt": prompt
    }
    response = requests.post(
        f'http://localhost:{PORT}/llm/paligemma',
        json=infer_payload,
    )
    resp = response.json()
    print(resp)


if __name__ == "__main__":
    do_gemma_request()