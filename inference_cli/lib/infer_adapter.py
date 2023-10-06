import base64

import requests


def is_url(string: str):
    return string.startswith("http")


def infer(image, project_id, model_version, api_key, host):
    endpoint = f"{host}/{project_id}/{model_version}"
    headers = {"Content-Type": "application/json"}

    if is_url(image):
        image_type = "url"
        params = {"api_key": api_key, "image": image, "image_type": image_type}
        response = requests.post(endpoint, headers=headers, params=params).json()
    else:
        with open(image, "rb") as image_file:
            image = base64.b64encode(image_file.read()).decode("utf-8")
        image_type = "base64"
        params = {"api_key": api_key, "image_type": image_type}
        response = requests.post(
            endpoint, headers=headers, params=params, data=image
        ).json()

    print(response)
