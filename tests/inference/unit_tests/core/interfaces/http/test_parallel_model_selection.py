import importlib
import sys
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from inference.core import env
from inference.core.interfaces.http import http_api


@pytest.fixture
def parallel_client(monkeypatch):
    tasks = ModuleType("inference.enterprise.parallel.tasks")
    setattr(tasks, "preprocess", MagicMock())
    monkeypatch.setitem(sys.modules, tasks.__name__, tasks)
    module_name = "inference.enterprise.parallel.dispatch_manager"
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    module = importlib.import_module(module_name)
    monkeypatch.setattr(env, "USE_INFERENCE_MODELS", True)
    monkeypatch.setattr(http_api, "InferenceInstrumentator", MagicMock())
    monkeypatch.setattr(http_api, "GCP_SERVERLESS", False)
    monkeypatch.setattr(http_api, "LAMBDA", False)
    monkeypatch.setattr(http_api, "DEDICATED_DEPLOYMENT_WORKSPACE_URL", None)
    monkeypatch.setattr(
        http_api.usage_collector, "async_push_usage_payloads", AsyncMock()
    )
    checker = MagicMock()
    manager = module.DispatchModelManager(
        MagicMock(), checker, content_addressed_artifact_cache=MagicMock()
    )
    try:
        with TestClient(http_api.HttpInterface(manager).app) as client:
            yield client, checker
    finally:
        sys.modules.pop(module_name, None)


@pytest.mark.parametrize(
    "path", ["/model/add", "/infer/object_detection", "/project/1"]
)
@pytest.mark.parametrize(
    "selectors",
    [{"backend": "onnx"}, {"quantization": "fp32"}, {"model_package_id": "package-1"}],
)
def test_parallel_routes_reject_selection_before_dispatch(
    parallel_client, path, selectors
):
    client, checker = parallel_client
    if path == "/project/1":
        response = client.post(
            path, params={"image": "https://example.com/image.jpg", **selectors}
        )
    else:
        response = client.post(
            path,
            json={
                "model_id": "project/1",
                "image": {"type": "url", "value": "https://example.com/image.jpg"},
                **selectors,
            },
        )
    assert response.status_code == 400
    assert "parallel mode" in response.json()["message"]
    checker.add_task.assert_not_called()
