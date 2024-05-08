import logging
import os
import time
from typing import Tuple, Optional

import pytest
import requests

logging.getLogger().setLevel(logging.WARNING)

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")


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
