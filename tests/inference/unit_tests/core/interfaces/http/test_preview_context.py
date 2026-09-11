import pytest

from inference_sdk.config import WORKFLOW_PREVIEW_HEADER, workflow_is_preview
from tests.inference.unit_tests.core.interfaces.http.test_workflow_sink_policy_contract import (
    _build_test_client,
)


@pytest.mark.parametrize(
    "header,expected", [("true", True), ("false", False), ("", False)]
)
def test_remote_preview_context_is_request_scoped(monkeypatch, header, expected):
    client = _build_test_client(monkeypatch)

    @client.app.get("/test-preview-context")
    async def read_preview():
        return {"is_preview": workflow_is_preview.get()}

    # Put the probe before the landing page catch-all mount.
    client.app.router.routes.insert(0, client.app.router.routes.pop())

    response = client.get(
        "/test-preview-context", headers={WORKFLOW_PREVIEW_HEADER: header}
    )
    assert response.status_code == 200
    assert response.json() == {"is_preview": expected}
    assert client.get("/test-preview-context").json() == {"is_preview": False}
    assert workflow_is_preview.get() is False
