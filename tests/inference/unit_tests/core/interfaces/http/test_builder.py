import json
import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient
from starlette.status import (
    HTTP_200_OK,
    HTTP_302_FOUND,
    HTTP_400_BAD_REQUEST,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_201_CREATED,
)

@pytest.fixture
def builder_app(builder_env_session):
    """
    This fixture runs AFTER the session-scoped fixture has set and reloaded
    the environment. We can now import our routes module fresh, and it will
    have created .csrf in the new tmp directory.
    """
    from inference.core.interfaces.http.builder import routes

    app = FastAPI()
    app.include_router(routes.router, prefix="/build")
    return app


def test_builder_html_injects_csrf(builder_app, builder_env_session):
    """
    The code in `routes` should have created `.csrf` in builder_env_session path.
    """
    client = TestClient(builder_app)
    response = client.get("/build")

    # The real code for your route returns 200 if we call /build (no slash).
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]

    # Make sure the .csrf file got created in the session fixture's path
    import os
    from pathlib import Path
    csrf_file = Path(builder_env_session) / "model_cache" / "workflow" / "local" / ".csrf"
    assert csrf_file.exists(), "CSRF file must exist"

    # Confirm the token is injected in the HTML:
    csrf_text = csrf_file.read_text()
    assert csrf_text in response.text


def test_builder_redirect_trailing_slash(builder_app):
    """
    Your code has two routes:
      @router.get("")
      @router.get("/", include_in_schema=False)
    That second route returns a 302. But *some* Starlette/ FastAPI combos
    might end up serving 200 or 302, depending on slash-handling.
    Let's see what the *actual* code does with /build/.
    """
    client = TestClient(builder_app)
    response = client.get("/build/", allow_redirects=False)

    # If your code actually returns 200, then do:
    #   assert response.status_code == 200
    #
    # But from your snippet, it looks like @router.get("/") returns a 302.
    # If that route is actually overshadowed by the no-slash route, you might get 200.
    # Let's test what happens in practice and adapt:
    assert response.status_code == HTTP_302_FOUND, f"Got {response.status_code} instead of 302"
    assert response.headers["location"] == "/build"


def test_builder_edit_injects_csrf(builder_app, builder_env_session):
    client = TestClient(builder_app)
    resp = client.get("/build/edit/my-workflow")
    assert resp.status_code == 200
    csrf_file = (
        Path(builder_env_session)
        / "model_cache"
        / "workflow"
        / "local"
        / ".csrf"
    )
    token = csrf_file.read_text()
    assert token in resp.text


# ---------------
# JSON API Tests
# ---------------

def test_api_get_all_workflows_unauthorized(builder_app):
    client = TestClient(builder_app)
    response = client.get("/build/api")
    assert response.status_code == HTTP_403_FORBIDDEN


def test_api_get_workflow_invalid_id(builder_app):
    """
    Your code returns 400 for invalid IDs.
    """
    client = TestClient(builder_app)
    # We'll need the csrf token to avoid 403
    from inference.core.interfaces.http.builder.routes import csrf
    response = client.get(
        "/build/api/../../etc/passwd",
        headers={"X-CSRF": csrf},
    )
    assert response.status_code == HTTP_400_BAD_REQUEST


def test_api_create_and_read(builder_app):
    """
    Demonstrate creating a valid workflow, then reading it.
    """
    client = TestClient(builder_app)
    from inference.core.interfaces.http.builder.routes import csrf

    # 1) create
    create_resp = client.post(
        "/build/api/test-wf",
        json={"id": "test-wf", "stuff": 123},
        headers={"X-CSRF": csrf},
    )
    assert create_resp.status_code == HTTP_201_CREATED

    # 2) get
    get_resp = client.get("/build/api/test-wf", headers={"X-CSRF": csrf})
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["data"]["config"] == {"id": "test-wf", "stuff": 123}


def test_fallback_redirect_invalid_id(builder_app):
    """
    Because the code's route is @router.get("/{workflow_id}", ...),
    and doesn't use {workflow_id:path}, "my/dir" won't match. 
    If the route doesn't match, you likely get 404. Let's see:
    """
    client = TestClient(builder_app)
    resp = client.get("/build/../../etc/passwd", allow_redirects=False)
    # The route doesn't match (it sees multiple segments),
    # so you actually get a 404. Let's confirm that:
    assert resp.status_code == 404, f"Expected 404, got {resp.status_code}"


def test_fallback_redirect_exists(builder_app):
    """
    If the file exists, code returns 302 -> /build/edit/{workflow_id}.
    So let's create the file on disk ourselves first,
    or we can do an API POST if that is simpler.
    """
    client = TestClient(builder_app)
    from inference.core.interfaces.http.builder.routes import csrf

    # Create file via the JSON API
    client.post(
        "/build/api/foobar",
        json={"id": "foobar"},
        headers={"X-CSRF": csrf},
    )

    # Now do GET /build/foobar
    resp = client.get("/build/foobar", allow_redirects=False)
    # The code is supposed to 302 to /build/edit/foobar if the file exists
    assert resp.status_code == HTTP_302_FOUND
    assert resp.headers["location"] == "/build/edit/foobar"


def test_fallback_redirect_not_exists(builder_app):
    """
    If file doesn't exist, returns 302 -> /build
    Actually, your code:
      if file_path.exists():
         302 -> /build/edit...
      else:
         302 -> /build
    Let's confirm that.
    """
    client = TestClient(builder_app)
    resp = client.get("/build/does-not-exist", allow_redirects=False)
    assert resp.status_code == HTTP_302_FOUND
    assert resp.headers["location"] == "/build"
