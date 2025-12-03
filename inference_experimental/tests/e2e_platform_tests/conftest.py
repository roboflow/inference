import os

import pytest


@pytest.fixture()
def roboflow_api_key() -> str:
    return os.environ["ROBOFLOW_API_KEY"]
