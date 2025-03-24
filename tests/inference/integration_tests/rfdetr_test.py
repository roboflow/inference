import os
import numpy as np
import pytest
import requests
from PIL import Image
from inference import get_model
from tests.inference.integration_tests.conftest import on_demand_clean_loaded_models
import time
import io
api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")

@pytest.fixture(scope="session", autouse=True)
def ensure_server_runs():
    try:
        res = requests.get(f"{base_url}:{port}")
        res.raise_for_status()
        success = True
    except:
        success = False
    max_wait = int(os.getenv("MAX_WAIT", 30))
    waited = 0
    while not success:
        print("Waiting for server to start...")
        time.sleep(5)
        waited += 5
        try:
            res = requests.get(f"{base_url}:{port}")
            res.raise_for_status()
            success = True
        except:
            success = False
        if waited > max_wait:
            raise Exception("Test server failed to start")

def test_rfdetr_base() -> None:
    on_demand_clean_loaded_models()
    response = requests.get("https://media.roboflow.com/dog.jpeg")
    response.raise_for_status()
    image = Image.open(io.BytesIO(response.content))
    model = get_model("rfdetr-base")
    payload = {
        "api_key": api_key,
        "image": {
            "type": "file",
            "value": image,
        },
        "confidence": 0.5,
    }

    predictions = model.infer(image, confidence=0.5)[0]

    assert predictions is not None, "Predictions should not be None"