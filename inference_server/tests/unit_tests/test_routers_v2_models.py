from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from inference_models.errors import ModelNotFoundError
from inference_server.framework import model_stat
from inference_server.framework.entities import CommonRequestParams
from inference_server.routers.v2_models import _interface_from_registry

_STAGE_TASK_TYPES = {
    "pp-ocrv6-det/small": "object-detection",
    "pp-ocrv6-rec/medium": "text-only-ocr",
}


@pytest.fixture(autouse=True)
def _reset():
    model_stat._reset_cache_for_tests()
    yield
    model_stat._reset_cache_for_tests()


def _registry(calls: list):
    def _metadata(model_id, api_key=None):
        calls.append((model_id, api_key))
        if model_id not in _STAGE_TASK_TYPES:
            raise ModelNotFoundError(message=model_id, help_url="")
        return MagicMock(task_type=_STAGE_TASK_TYPES[model_id])

    return patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=_metadata,
    )


@pytest.mark.asyncio
async def test_interface_of_a_pipeline_id_resolves_without_statting_it():
    calls: list = []
    with _registry(calls):
        response = await _interface_from_registry(
            CommonRequestParams(model_id="pp_ocr/small-medium", api_key="key-1")
        )
    assert response is not None and response.status_code == 200
    body = json.loads(response.body)
    assert body["model_id"] == "pp_ocr/small-medium"
    assert body["model_type"] == "structured-ocr"
    assert "infer" in body["actions"]
    assert sorted(calls) == [
        ("pp-ocrv6-det/small", "key-1"),
        ("pp-ocrv6-rec/medium", "key-1"),
    ]


@pytest.mark.asyncio
async def test_interface_of_a_pipeline_id_with_a_denied_stage_is_401():
    from inference_models.errors import UnauthorizedModelAccessError

    def _metadata(model_id, api_key=None):
        if model_id == "pp-ocrv6-rec/medium":
            raise UnauthorizedModelAccessError(message=model_id, help_url="")
        return MagicMock(task_type="object-detection")

    with patch(
        "inference_server.framework.model_stat.get_one_page_of_model_metadata",
        side_effect=_metadata,
    ):
        response = await _interface_from_registry(
            CommonRequestParams(model_id="pp_ocr/small-medium", api_key="key-1")
        )
    assert response is not None and response.status_code == 401
