import base64
import os
import time
from io import BytesIO

import pytest
import requests
from PIL import Image

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")

MINIMUM_FPS = int(os.environ.get("MINIMUM_FPS", 1))


def bool_env(val):
    if isinstance(val, bool):
        return val
    return val.lower() in ["true", "1", "t", "y", "yes"]


def model_add(test, port=9001, api_key="", base_url="http://localhost"):
    return requests.post(
        f"{base_url}:{port}/{test['project']}/{test['version']}?"
        + "&".join(
            [
                f"api_key={api_key}",
                f"confidence={test['confidence']}",
                f"overlap={test['iou_threshold']}",
                f"image={test['image_url']}",
            ]
        )
    )


@pytest.mark.skipif(
    bool_env(os.getenv("SKIP_SPEED_TEST", False)), reason="Skipping speed test"
)
def test_speed(clean_loaded_models_fixture):
    try:
        buffered = BytesIO()
        image_url = "https://source.roboflow.com/D8zLgnZxdqtqF0plJINA/DqK7I0rUz5HBvu1hdNi6/original.jpg"
        pil_image = Image.open(requests.get(image_url, stream=True).raw)
        pil_image.save(buffered, quality=100, format="PNG")
        img_str = base64.b64encode(buffered.getvalue())
        img_str = img_str.decode("ascii")
        times = []
        errors = 0
        for _ in range(250):
            start = time.time()
            res = requests.post(
                f"{base_url}:{port}/coco/3?"
                + "&".join(
                    [
                        f"api_key={api_key}",
                        f"confidence=0.5",
                        f"overlap=0.5",
                    ]
                ),
                data=img_str,
                headers={"Content-Type": "application/json"},
            )
            try:
                res.raise_for_status()
                times.append(time.time() - start)
            except:
                errors += 1
                if errors > 5:
                    raise Exception("Too many errors during speed test")
        times = times[10:]
        avg = sum(times) / len(times)
        fps = 1 / avg
        print(f"Average FPS: {fps}")
        if fps < MINIMUM_FPS:
            raise Exception(f"FPS too low: {fps} < {MINIMUM_FPS}")
    except Exception as e:
        raise Exception(f"Error in speed test: {e}")


@pytest.fixture(scope="session", autouse=True)
def setup():
    try:
        res = requests.get(f"{base_url}:{port}")
        res.raise_for_status()
        success = True
    except:
        success = False
    MAX_WAIT = int(os.getenv("MAX_WAIT", 30))
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
        if waited > MAX_WAIT:
            raise Exception("Test server failed to start")
