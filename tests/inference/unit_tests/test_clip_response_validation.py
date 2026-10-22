from contextlib import nullcontext
from copy import deepcopy
from unittest.mock import Mock

import pytest
import requests

from tests.inference.integration_tests import clip_test
from tests.inference.integration_tests.conftest import API_KEY_AUTH_MODES


@pytest.fixture
def _response(monkeypatch):
    response = Mock(spec=requests.Response, status_code=200)
    monkeypatch.setattr(clip_test, "api_key", None)
    monkeypatch.setattr(clip_test.requests, "post", Mock(return_value=response))
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        Mock(side_effect=RuntimeError("Unexpected HTTP request")),
    )
    return response


@pytest.mark.parametrize("case", clip_test.TESTS, ids=lambda case: case["description"])
@pytest.mark.parametrize("auth_mode", API_KEY_AUTH_MODES)
@pytest.mark.parametrize("functional", [False, True])
@pytest.mark.parametrize("response_kind", ["valid", "wrong", "missing"])
def test_clip_response_validation(
    _response, monkeypatch, case, auth_mode, functional, response_kind
):
    """Check the existing CLIP test against offline responses.

    Args:
        _response: Mock HTTP response with successful status checking.
        monkeypatch: Pytest patch manager for the FUNCTIONAL setting.
        case: Existing CLIP request and expected response.
        auth_mode: API-key transport used by the integration suite.
        functional: Whether only HTTP success should be checked.
        response_kind: Valid data, a wrong result, or a missing result field.
    """
    monkeypatch.setenv("FUNCTIONAL", str(functional))
    data = deepcopy(case["expected_response"])
    field = "similarity" if case["type"] == "compare" else "embeddings"
    if response_kind == "wrong":
        data[field] = [-1] if field == "similarity" else []
    elif response_kind == "missing":
        del data[field]
    _response.json.return_value = data

    expectation = (
        pytest.raises(AssertionError)
        if not functional and response_kind != "valid"
        else nullcontext()
    )
    with expectation:
        clip_test.test_clip(
            test=case, auth_mode=auth_mode, clean_loaded_models_fixture=None
        )

    _response.raise_for_status.assert_called_once_with()
    if functional:
        _response.json.assert_not_called()
    else:
        _response.json.assert_called_once_with()


@pytest.mark.parametrize("functional", [False, True])
def test_clip_http_errors(_response, monkeypatch, functional):
    """Require HTTP errors to fail in both modes.

    Args:
        _response: Mock HTTP response that raises an HTTP error.
        monkeypatch: Pytest patch manager for the FUNCTIONAL setting.
        functional: Whether only HTTP success should be checked.
    """
    monkeypatch.setenv("FUNCTIONAL", str(functional))
    _response.raise_for_status.side_effect = requests.HTTPError("HTTP 500")

    with pytest.raises(requests.HTTPError):
        clip_test.test_clip(
            test=clip_test.TESTS[0],
            auth_mode="legacy",
            clean_loaded_models_fixture=None,
        )

    _response.json.assert_not_called()
