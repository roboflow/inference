import logging
import os
import time
from typing import Dict, List, Optional, Tuple

import pytest
import requests

logging.getLogger().setLevel(logging.WARNING)

api_key = os.environ.get("API_KEY")
port = os.environ.get("PORT", 9001)
base_url = os.environ.get("BASE_URL", "http://localhost")

print(base_url, port)

# The two API-key transports every authenticated test runs under. Trimming this
# list back to ["legacy"] is the single switch disabling the header-auth lane.
API_KEY_AUTH_MODES = ["legacy", "header"]


@pytest.fixture(params=API_KEY_AUTH_MODES)
def auth_mode(request) -> str:
    """Duplicate a test over the two API-key transports.

    "legacy" - api_key travels in the query string / JSON body, byte-identical
    to how the tests always sent it. "header" - api_key is stripped from
    query/body and travels as `Authorization: Bearer <api_key>` instead.
    """
    return request.param


def api_key_auth_headers(auth_mode: str, api_key: Optional[str]) -> Dict[str, str]:
    """Return headers carrying the api key - non-empty only in "header" mode."""
    if auth_mode == "header" and api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def without_api_key_in_header_mode(
    auth_mode: str, payload: Optional[dict]
) -> Optional[dict]:
    """Strip `api_key` from a JSON payload / query-params dict in "header" mode.

    In "legacy" mode the payload is returned unchanged (same object), keeping
    the wire bytes identical to the pre-dual-mode tests.
    """
    if auth_mode != "header" or payload is None:
        return payload
    return {key: value for key, value in payload.items() if key != "api_key"}


def api_key_query_fragments(auth_mode: str, api_key: Optional[str]) -> List[str]:
    """`api_key=...` fragments for hand-assembled query strings ("legacy" only)."""
    if auth_mode == "header":
        return []
    return [f"api_key={api_key}"]


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
