import os
import json
import pytest
import re
from pathlib import Path
from starlette.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_302_FOUND,
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
)

# We'll assume you can import the 'router' directly, which is your "builder" routes
# from inference.core.interfaces.http.builder.routes import router

# But since we need to override environment variables and re-import, we might do:
from fastapi import FastAPI

@pytest.fixture
def builder_app(tmp_path, monkeypatch):
    """
    This fixture:
      1. Points MODEL_CACHE_DIR to a temporary directory
      2. Imports (or re-imports) the router to ensure the .csrf file is generated in tmp_path
      3. Mounts that router onto a FastAPI app
    """
    # 1) Point environment variable to temporary directory
    #    This ensures your 'workflow_local_dir' is created in the temp dir
    monkeypatch.setenv("MODEL_CACHE_DIR", str(tmp_path / "model_cache"))

    # 2) Because your module-level code runs once at import time, we re-import it
    #    so that it picks up the new environment variable and creates the .csrf in the tmp dir.
    #    This depends on your exact project structure, so adjust as needed.
    from inference.core.interfaces.http.builder import routes as builder_routes

    # 3) Create a minimal FastAPI (or Starlette) app, and include the router
    app = FastAPI()
    app.include_router(builder_routes.router, prefix="/build")
    return app


def test_builder_html_injects_csrf(builder_app, tmp_path):
    """
    Test that the GET /build route returns HTML containing the CSRF token.
    """
    client = TestClient(builder_app)
    response = client.get("/build")
    assert response.status_code == HTTP_200_OK
    assert response.headers["content-type"].startswith("text/html")

    # The .csrf file should exist in tmp_path
    csrf_file = tmp_path / "model_cache" / "workflow" / "local" / ".csrf"
    assert csrf_file.exists(), "CSRF file must exist"
    token = csrf_file.read_text()

    # The HTML should contain the CSRF token
    assert token in response.text, "HTML must contain the CSRF token"


def test_builder_redirect_trailing_slash(builder_app):
    """
    Test that GET /build/ (with trailing slash) returns 302 -> /build
    """
    client = TestClient(builder_app)
    response = client.get("/build/")
    assert response.status_code == HTTP_302_FOUND
    assert response.headers["location"] == "/build"


def test_builder_edit_injects_csrf(builder_app, tmp_path):
    """
    Test that GET /build/edit/{workflow_id} returns HTML containing the CSRF token.
    """
    client = TestClient(builder_app)
    response = client.get("/build/edit/some-workflow")
    assert response.status_code == HTTP_200_OK
    csrf_file = tmp_path / "model_cache" / "workflow" / "local" / ".csrf"
    token = csrf_file.read_text()
    assert token in response.text, "HTML must contain the CSRF token"


# ---------------
# JSON API Tests
# ---------------

@pytest.fixture
def get_csrf_token(tmp_path):
    """
    Returns the actual CSRF token from .csrf file in the tmp_path.
    """
    def _inner():
        csrf_file = tmp_path / "model_cache" / "workflow" / "local" / ".csrf"
        return csrf_file.read_text()
    return _inner


def test_api_get_all_workflows_unauthorized(builder_app):
    """
    GET /build/api with no X-CSRF header should fail with 403.
    """
    client = TestClient(builder_app)
    response = client.get("/build/api")
    assert response.status_code == HTTP_403_FORBIDDEN


def test_api_get_all_workflows_empty(builder_app, get_csrf_token):
    """
    GET /build/api with correct X-CSRF token but no workflows present -> returns empty data.
    """
    client = TestClient(builder_app)
    csrf = get_csrf_token()
    response = client.get("/build/api", headers={"X-CSRF": csrf})
    assert response.status_code == HTTP_200_OK
    result = response.json()
    assert result["data"] == {}, "Should return empty data dict when no workflows are present"


def test_api_get_workflow_invalid_id(builder_app, get_csrf_token):
    """
    GET /build/api/../../etc/passwd to ensure the regex blocks invalid IDs.
    Should return 400 instead of reading arbitrary files.
    """
    client = TestClient(builder_app)
    csrf = get_csrf_token()
    malicious_id = "../../etc/passwd"

    response = client.get(f"/build/api/{malicious_id}", headers={"X-CSRF": csrf})
    assert response.status_code == HTTP_400_BAD_REQUEST
    result = response.json()
    assert result["error"] == "invalid id"


def test_api_create_workflow_invalid_id(builder_app, get_csrf_token):
    """
    POST /build/api/<invalid> should fail with 400.
    """
    client = TestClient(builder_app)
    csrf = get_csrf_token()

    response = client.post(
        "/build/api/../../etc/passwd",
        headers={"X-CSRF": csrf},
        json={"data": "blah"},
    )
    assert response.status_code == HTTP_400_BAD_REQUEST
    result = response.json()
    assert result["error"] == "invalid id"


def test_api_create_and_get_workflow(builder_app, get_csrf_token, tmp_path):
    """
    1) POST /build/api/my-workflow -> create file
    2) GET /build/api/my-workflow -> read it back
    3) verify the JSON content
    """
    client = TestClient(builder_app)
    csrf = get_csrf_token()

    create_resp = client.post(
        "/build/api/my-workflow",
        headers={"X-CSRF": csrf},
        json={"id": "my-workflow", "stuff": 123},
    )
    assert create_resp.status_code == HTTP_201_CREATED
    assert create_resp.json()["message"] == "Workflow 'my-workflow' created/updated successfully."

    # Check the file on disk
    file_path = tmp_path / "model_cache" / "workflow" / "local" / "my-workflow.json"
    assert file_path.exists()
    file_contents = json.loads(file_path.read_text())
    assert file_contents == {"id": "my-workflow", "stuff": 123}

    # Now GET it
    get_resp = client.get("/build/api/my-workflow", headers={"X-CSRF": csrf})
    assert get_resp.status_code == HTTP_200_OK
    data = get_resp.json()
    assert "data" in data
    # The "config" should match the file contents we wrote
    assert data["data"]["config"] == {"id": "my-workflow", "stuff": 123}


def test_api_rename_workflow(builder_app, get_csrf_token, tmp_path):
    """
    If the JSON body has an 'id' different from the URL param, the code tries to rename
    (i.e. delete the old file) and rewrite under the new name. We verify that logic.
    """
    client = TestClient(builder_app)
    csrf = get_csrf_token()

    # 1) Create a workflow with ID 'old-workflow'
    create_resp = client.post(
        "/build/api/old-workflow",
        headers={"X-CSRF": csrf},
        json={"id": "old-workflow", "stuff": 111},
    )
    assert create_resp.status_code == HTTP_201_CREATED

    old_file = tmp_path / "model_cache" / "workflow" / "local" / "old-workflow.json"
    new_file = tmp_path / "model_cache" / "workflow" / "local" / "new-workflow.json"
    assert old_file.exists()
    assert not new_file.exists()

    # 2) POST to /build/api/new-workflow, but body contains "id=old-workflow"
    #    The code checks mismatch, attempts to delete old-workflow file, and writes new-workflow.
    rename_resp = client.post(
        "/build/api/new-workflow",
        headers={"X-CSRF": csrf},
        json={"id": "old-workflow", "stuff": 222},
    )
    assert rename_resp.status_code == HTTP_201_CREATED

    # old file should be removed, new file created
    assert not old_file.exists()
    assert new_file.exists()
    contents = json.loads(new_file.read_text())
    # The 'id' in the new file should be "new-workflow"
    assert contents["id"] == "new-workflow"
    assert contents["stuff"] == 222


def test_api_delete_workflow(builder_app, get_csrf_token, tmp_path):
    """
    1) Create a workflow
    2) Delete it
    3) Confirm 404 if we try to GET or DELETE again
    """
    client = TestClient(builder_app)
    csrf = get_csrf_token()

    # Create
    client.post(
        "/build/api/delete-me",
        headers={"X-CSRF": csrf},
        json={"id": "delete-me", "stuff": 777},
    )
    file_path = tmp_path / "model_cache" / "workflow" / "local" / "delete-me.json"
    assert file_path.exists()

    # Delete
    del_resp = client.delete("/build/api/delete-me", headers={"X-CSRF": csrf})
    assert del_resp.status_code == HTTP_200_OK
    assert del_resp.json()["message"] == "Workflow 'delete-me' deleted successfully."
    assert not file_path.exists(), "File should be removed after DELETE"

    # GET again -> 404
    get_resp = client.get("/build/api/delete-me", headers={"X-CSRF": csrf})
    assert get_resp.status_code == HTTP_404_NOT_FOUND
    # DELETE again -> 404
    del_resp_2 = client.delete("/build/api/delete-me", headers={"X-CSRF": csrf})
    assert del_resp_2.status_code == HTTP_404_NOT_FOUND


# -------------------------
# Fallback redirect route
# -------------------------

def test_fallback_redirect_invalid_id(builder_app):
    """
    GET /build/<workflow_id> with an invalid workflow_id -> 302 redirect to /build
    """
    client = TestClient(builder_app)
    response = client.get("/build/../../etc/passwd")
    assert response.status_code == HTTP_302_FOUND
    assert response.headers["location"] == "/build"


def test_fallback_redirect_exists(builder_app, get_csrf_token, tmp_path):
    """
    If the workflow file exists, /build/<workflow_id> -> /build/edit/<workflow_id>
    """
    client = TestClient(builder_app)
    csrf = get_csrf_token()

    # Create a file
    client.post(
        "/build/api/fallback-test",
        headers={"X-CSRF": csrf},
        json={"id": "fallback-test"},
    )
    # Now the file "fallback-test.json" exists

    response = client.get("/build/fallback-test")
    assert response.status_code == HTTP_302_FOUND
    assert response.headers["location"] == "/build/edit/fallback-test"


def test_fallback_redirect_not_exists(builder_app):
    """
    If the workflow file doesn't exist, /build/<workflow_id> -> /build
    """
    client = TestClient(builder_app)
    response = client.get("/build/does-not-exist")
    assert response.status_code == HTTP_302_FOUND
    assert response.headers["location"] == "/build"
