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
def builder_app(builder_env_session):
    """
    After the environment is set via builder_env_session, import the routes module.
    """
    from inference.core.interfaces.http.builder import routes

    app = FastAPI()
    app.include_router(routes.router, prefix="/build")
    return app


def test_builder_html_injects_csrf(builder_app, builder_env_session):
    """
    Instead of checking for a file, we verify that the HTML response
    contains a valid CSRF token (32 hex digits).
    """
    client = TestClient(builder_app)
    response = client.get("/build")
    assert response.status_code == HTTP_200_OK
    assert "text/html" in response.headers["content-type"]

    # Extract CSRF token from the HTML response.
    token_match = re.search(r"CSRF:\s*([0-9a-f]+)", response.text)
    assert token_match, "CSRF token not found in HTML response"
    token = token_match.group(1)
    assert len(token) == 32, "CSRF token should be 32 hex digits long"


def test_builder_redirect_trailing_slash(builder_app):
    """
    Verify that GET /build/ returns a redirect.
    (Note: we use follow_redirects=False and the new keyword `follow_redirects`
    if needed to avoid deprecation warnings.)
    """
    client = TestClient(builder_app)
    response = client.get("/build/", follow_redirects=False)
    assert response.status_code == HTTP_302_FOUND, f"Expected 302, got {response.status_code}"
    assert response.headers["location"] == "/build"


def test_builder_edit_injects_csrf(builder_app, builder_env_session):
    from pathlib import Path  # ensure Path is imported
    client = TestClient(builder_app)
    response = client.get("/build/edit/my-workflow")
    assert response.status_code == HTTP_200_OK

    # Verify that the HTML contains a CSRF token.
    token_match = re.search(r"CSRF:\s*([0-9a-f]+)", response.text)
    assert token_match, "CSRF token not found in HTML response"
    token = token_match.group(1)
    assert len(token) == 32, "CSRF token should be 32 hex digits long"


# ---------------
# JSON API Tests
# ---------------

def test_api_get_all_workflows_unauthorized(builder_app):
    client = TestClient(builder_app)
    response = client.get("/build/api")
    assert response.status_code == HTTP_403_FORBIDDEN


def test_api_get_workflow_invalid_id(builder_app):
    """
    Use an invalid workflow_id that is syntactically invalid (contains a '$')
    so that the regex check in the route triggers and returns 400.
    """
    client = TestClient(builder_app)
    # Get the CSRF token from the routes module
    from inference.core.interfaces.http.builder.routes import csrf
    invalid_id = "invalid$id"  # '$' is not allowed by regex [\w\-]+
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

    # Get the workflow
    get_resp = client.get("/build/api/test-wf", headers={"X-CSRF": csrf})
    assert get_resp.status_code == HTTP_200_OK
    data = get_resp.json()
    assert data["data"]["config"] == {"id": "test-wf", "stuff": 123}


def test_fallback_redirect_invalid_id(builder_app):
    """
    With an invalid id that contains slashes, the route will not match,
    leading to a 404.
    """
    client = TestClient(builder_app)
    response = client.get("/build/../../etc/passwd", follow_redirects=False)
    assert response.status_code == HTTP_404_NOT_FOUND, f"Expected 404, got {response.status_code}"


def test_fallback_redirect_exists(builder_app):
    """
    Create a workflow file via the JSON API and then verify that GET /build/<id>
    redirects to /build/edit/<id>.
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
