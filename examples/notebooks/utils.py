import getpass

import cv2
import numpy as np
import requests

from inference.core.env import API_KEY


def get_roboflow_api_key():
    if API_KEY is None:
        api_key = getpass.getpass("Roboflow API Key:")
    else:
        api_key = API_KEY
    return api_key

def load_image_from_url(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Ensure that the request was successful
    if response.status_code == 200:
        # Convert the response content into a numpy array
        image_array = np.asarray(bytearray(response.content), dtype=np.uint8)

        # Decode the image array into an OpenCV image
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        return image
    else:
        print(f"Failed to retrieve the image. HTTP status code: {response.status_code}")
        return None