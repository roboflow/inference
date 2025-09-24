import logging
import os
import time
from typing import Optional, Tuple

import pytest
import requests

logging.getLogger().setLevel(logging.WARNING)

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")

print(base_url, port)

@pytest.fixture(scope="session", autouse=True)
def server_url() -> str:
    # TODO: start using everywhere
    server_url = f"{base_url}:{port}"
    try:
        res = requests.get(server_url)
        res.raise_for_status()
        success = True
    except:
        success = False
    max_wait = int(os.getenv("MAX_WAIT", 30))
    waited = 0
    while not success:
        if waited > max_wait:
            raise TimeoutError("Test server failed to start")
        logging.warning("Waiting for server to start...")
        time.sleep(5)
        waited += 5
        try:
            res = requests.get(server_url)
            res.raise_for_status()
            success = True
        except:
            success = False
    return server_url


@pytest.fixture(scope="module")
def clean_loaded_models_fixture() -> None:
    on_demand_clean_loaded_models()


@pytest.fixture()
def clean_loaded_models_every_test_fixture() -> None:
    on_demand_clean_loaded_models()


def on_demand_clean_loaded_models() -> None:
    response = requests.post(f"{base_url}:{port}/model/clear")
    response.raise_for_status()
