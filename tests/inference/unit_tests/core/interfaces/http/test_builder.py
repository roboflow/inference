import json
import re
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_302_FOUND,
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
)
from starlette.testclient import TestClient


@pytest.fixture
def builder_app(builder_env_session, monkeypatch):
    """
    Import the routes module after the environment is set.
    Monkeypatch Path.read_text so that when the routes read "editor.html",
    a predictable template is returned.
    """
    from inference.core.interfaces.http.builder import routes

    # Save the original read_text method
    original_read_text = Path.read_text

    def fake_read_text(self, encoding="utf-8"):
        if self.name == "editor.html":
            # Return a test template that includes a placeholder for the CSRF token.
            return "Test Editor HTML: CSRF={{CSRF}}"
        return original_read_text(self, encoding=encoding)

    monkeypatch.setattr(Path, "read_text", fake_read_text)

    app = FastAPI()
    app.include_router(routes.router, prefix="/build")
    return app


def test_builder_html_injects_csrf(builder_app, builder_env_session):
    """
    Verify that the HTML response for GET /build has the CSRF token injected.
    Our fake template returns "Test Editor HTML: CSRF={{CSRF}}", so after
    replacement it should contain "CSRF=<actual_token>".
    """
    client = TestClient(builder_app)
    response = client.get("/build")
    assert response.status_code == HTTP_200_OK
    assert "text/html" in response.headers["content-type"]

    # Look for the CSRF token injected into the template.
    token_match = re.search(r"CSRF=([0-9a-f]+)", response.text)
    assert token_match, "CSRF token not found in HTML response"
    token = token_match.group(1)
    assert len(token) == 32, "CSRF token should be 32 hex digits long"


def test_builder_edit_injects_csrf(builder_app, builder_env_session):
    """
    Verify that GET /build/edit/{workflow_id} returns HTML with the CSRF token.
    """
    client = TestClient(builder_app)
    response = client.get("/build/edit/my-workflow")
    assert response.status_code == HTTP_200_OK
    token_match = re.search(r"CSRF=([0-9a-f]+)", response.text)
    assert token_match, "CSRF token not found in HTML response"
    token = token_match.group(1)
    assert len(token) == 32, "CSRF token should be 32 hex digits long"


def test_builder_redirect_trailing_slash(builder_app):
    """
    Verify that GET /build/ returns a redirect.
    """
    client = TestClient(builder_app)
    response = client.get("/build/", follow_redirects=False)
    assert (
        response.status_code == HTTP_302_FOUND
    ), f"Expected 302, got {response.status_code}"
    assert response.headers["location"] == "/build"


def test_api_get_all_workflows_unauthorized(builder_app):
    client = TestClient(builder_app)
    response = client.get("/build/api")
    assert response.status_code == HTTP_403_FORBIDDEN


def test_api_get_workflow_invalid_id(builder_app):
    """
    Use an invalid workflow_id (with an illegal character) so that the route
    returns 400 instead of 404. (Using slashes would fail to match the route.)
    """
    client = TestClient(builder_app)
    from inference.core.interfaces.http.builder.routes import csrf

    invalid_id = "invalid$id"  # '$' is not allowed by the regex [\w\-]+
    response = client.get(
        f"/build/api/{invalid_id}",
        headers={"X-CSRF": csrf},
    )
    assert response.status_code == HTTP_400_BAD_REQUEST


def test_api_create_and_read(builder_app):
    client = TestClient(builder_app)
    from inference.core.interfaces.http.builder.routes import csrf

    # Create a workflow
    create_resp = client.post(
        "/build/api/test-wf",
        json={"id": "test-wf", "stuff": 123},
        headers={"X-CSRF": csrf},
    )
    assert create_resp.status_code == HTTP_201_CREATED

    # Read the created workflow
    get_resp = client.get("/build/api/test-wf", headers={"X-CSRF": csrf})
    assert get_resp.status_code == HTTP_200_OK
    data = get_resp.json()
    assert data["data"]["config"] == {"id": "test-wf", "stuff": 123}


def test_fallback_redirect_invalid_id(builder_app):
    """
    With an invalid id containing slashes, the route will not match and yield a 404.
    """
    client = TestClient(builder_app)
    response = client.get("/build/../../etc/passwd", follow_redirects=False)
    assert (
        response.status_code == HTTP_404_NOT_FOUND
    ), f"Expected 404, got {response.status_code}"


def test_fallback_redirect_exists(builder_app):
    """
    Create a workflow via the JSON API and verify that GET /build/<id> redirects to /build/edit/<id>.
    """
    client = TestClient(builder_app)
    from inference.core.interfaces.http.builder.routes import csrf

    client.post(
        "/build/api/foobar",
        json={"id": "foobar"},
        headers={"X-CSRF": csrf},
    )
    response = client.get("/build/foobar", follow_redirects=False)
    assert response.status_code == HTTP_302_FOUND
    assert response.headers["location"] == "/build/edit/foobar"


def test_fallback_redirect_not_exists(builder_app):
    """
    If the workflow file does not exist, GET /build/<id> should redirect to /build.
    """
    client = TestClient(builder_app)
    response = client.get("/build/does-not-exist", follow_redirects=False)
    assert response.status_code == HTTP_302_FOUND
    assert response.headers["location"] == "/build"
