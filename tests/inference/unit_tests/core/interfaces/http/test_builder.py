import json
import logging
import os
import re
import subprocess
import sys
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
    HTTP_500_INTERNAL_SERVER_ERROR,
)
from starlette.testclient import TestClient


def _workflow_file_path(routes, workflow_id: str) -> Path:
    workflow_hash = routes.sha256(workflow_id.encode()).hexdigest()
    return routes.workflow_local_dir / f"{workflow_hash}.json"


def _install_final_workflow_symlink(routes, tmp_path: Path, workflow_id: str):
    outside_file = tmp_path / f"{workflow_id}-outside.json"
    outside_contents = {"id": workflow_id, "source": "outside-cache-root"}
    outside_file.write_text(json.dumps(outside_contents))
    workflow_file = _workflow_file_path(routes=routes, workflow_id=workflow_id)
    workflow_file.symlink_to(outside_file)
    return workflow_file, outside_file, outside_contents


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


def test_api_create_rejects_final_workflow_symlink(builder_app, tmp_path):
    from inference.core.interfaces.http.builder import routes

    workflow_id = "write-symlink"
    workflow_file, outside_file, outside_contents = (
        _install_final_workflow_symlink(
            routes=routes,
            tmp_path=tmp_path,
            workflow_id=workflow_id,
        )
    )
    client = TestClient(builder_app)

    response = client.post(
        f"/build/api/{workflow_id}",
        json={"id": workflow_id, "source": "replacement"},
        headers={"X-CSRF": routes.csrf},
    )

    assert response.status_code == HTTP_500_INTERNAL_SERVER_ERROR
    assert response.json() == {"error": "unsafe cache path"}
    assert workflow_file.is_symlink()
    assert json.loads(outside_file.read_text()) == outside_contents


def test_api_get_rejects_final_workflow_symlink(builder_app, tmp_path):
    from inference.core.interfaces.http.builder import routes

    workflow_id = "read-symlink"
    workflow_file, outside_file, outside_contents = (
        _install_final_workflow_symlink(
            routes=routes,
            tmp_path=tmp_path,
            workflow_id=workflow_id,
        )
    )
    client = TestClient(builder_app)

    response = client.get(
        f"/build/api/{workflow_id}",
        headers={"X-CSRF": routes.csrf},
    )

    assert response.status_code == HTTP_404_NOT_FOUND
    assert workflow_file.is_symlink()
    assert json.loads(outside_file.read_text()) == outside_contents


def test_api_delete_rejects_final_workflow_symlink(builder_app, tmp_path):
    from inference.core.interfaces.http.builder import routes

    workflow_id = "delete-symlink"
    workflow_file, outside_file, outside_contents = (
        _install_final_workflow_symlink(
            routes=routes,
            tmp_path=tmp_path,
            workflow_id=workflow_id,
        )
    )
    client = TestClient(builder_app)

    response = client.delete(
        f"/build/api/{workflow_id}",
        headers={"X-CSRF": routes.csrf},
    )

    assert response.status_code == HTTP_404_NOT_FOUND
    assert workflow_file.is_symlink()
    assert json.loads(outside_file.read_text()) == outside_contents


def test_api_list_skips_final_workflow_symlink(builder_app, tmp_path):
    from inference.core.interfaces.http.builder import routes

    workflow_id = "list-symlink"
    workflow_file, outside_file, outside_contents = (
        _install_final_workflow_symlink(
            routes=routes,
            tmp_path=tmp_path,
            workflow_id=workflow_id,
        )
    )
    client = TestClient(builder_app)

    response = client.get(
        "/build/api",
        headers={"X-CSRF": routes.csrf},
    )

    assert response.status_code == HTTP_200_OK
    assert workflow_id not in response.json()["data"]
    assert workflow_file.is_symlink()
    assert json.loads(outside_file.read_text()) == outside_contents


def test_api_overwrite_replaces_workflow_atomically(
    builder_app,
    monkeypatch,
):
    from inference.core.interfaces.http.builder import routes

    workflow_id = "atomic-overwrite"
    workflow_file = _workflow_file_path(routes=routes, workflow_id=workflow_id)
    client = TestClient(builder_app)
    first_response = client.post(
        f"/build/api/{workflow_id}",
        json={"id": workflow_id, "revision": "old"},
        headers={"X-CSRF": routes.csrf},
    )
    assert first_response.status_code == HTTP_201_CREATED

    replace_observations = []
    original_replace = routes.os.replace

    def inspect_atomic_replace(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        replace_observations.append(
            {
                "source": json.loads(source_path.read_text()),
                "destination": json.loads(destination_path.read_text()),
                "temporary_name": source_path.name,
            }
        )
        original_replace(source, destination)

    monkeypatch.setattr(routes.os, "replace", inspect_atomic_replace)

    second_response = client.post(
        f"/build/api/{workflow_id}",
        json={"id": workflow_id, "revision": "new"},
        headers={"X-CSRF": routes.csrf},
    )

    assert second_response.status_code == HTTP_201_CREATED
    assert len(replace_observations) == 1
    replace_observation = replace_observations[0]
    assert replace_observation["source"] == {
        "id": workflow_id,
        "revision": "new",
    }
    assert replace_observation["destination"] == {
        "id": workflow_id,
        "revision": "old",
    }
    assert replace_observation["temporary_name"].startswith(".local-workflow.")
    assert json.loads(workflow_file.read_text()) == {
        "id": workflow_id,
        "revision": "new",
    }
    assert list(routes.workflow_local_dir.glob(".local-workflow.*.tmp")) == []


def test_cached_model_aggregation_excludes_conflicting_metadata(
    builder_app,
    caplog,
    monkeypatch,
):
    from inference.core.interfaces.http.builder import routes

    shared_model_first_root = {
        "model_id": "workspace/shared/1",
        "name": "workspace/shared/1",
        "task_type": "object-detection",
        "model_architecture": "architecture-a",
        "is_foundation": False,
    }
    shared_model_second_root = {
        **shared_model_first_root,
        "task_type": "classification",
        "model_architecture": "architecture-b",
    }
    identical_model = {
        "model_id": "workspace/identical/1",
        "name": "workspace/identical/1",
        "task_type": "object-detection",
        "model_architecture": "architecture-c",
        "is_foundation": False,
    }
    unique_model = {
        "model_id": "workspace/unique/1",
        "name": "workspace/unique/1",
        "task_type": "classification",
        "model_architecture": "architecture-d",
        "is_foundation": False,
    }

    models_by_root = {
        "/cache/first": [
            shared_model_first_root,
            identical_model,
            unique_model,
        ],
        "/cache/second": [
            shared_model_second_root,
            dict(identical_model),
            dict(shared_model_first_root),
        ],
    }

    def scan_cache_root(cache_root, excluded_cache_roots):
        return models_by_root[cache_root]

    fake_aliases_module = type(sys)("inference.models.aliases")
    fake_aliases_module.REGISTERED_ALIASES = {}
    monkeypatch.setitem(
        sys.modules,
        "inference.models.aliases",
        fake_aliases_module,
    )
    monkeypatch.setattr(
        routes,
        "get_configured_model_cache_roots",
        lambda: list(models_by_root),
    )
    monkeypatch.setattr(routes, "scan_cached_models", scan_cache_root)
    monkeypatch.setattr(routes, "load_workflow_blocks", lambda: [])
    monkeypatch.setattr(
        routes,
        "get_cached_foundation_models",
        lambda blocks: [],
    )
    monkeypatch.setattr(
        routes,
        "get_task_type_to_block_mapping",
        lambda blocks: {},
    )
    monkeypatch.setattr(routes, "_models_cache", None)
    client = TestClient(builder_app)

    with caplog.at_level(logging.WARNING, logger=routes.__name__):
        response = client.get(
            "/build/api/models",
            headers={"X-CSRF": routes.csrf},
        )

    assert response.status_code == HTTP_200_OK
    models_by_id = {
        model["model_id"]: model for model in response.json()["models"]
    }
    assert set(models_by_id) == {
        identical_model["model_id"],
        unique_model["model_id"],
    }
    assert caplog.messages == [
        "Excluding cached model workspace/shared/1 because configured cache "
        "roots contain conflicting metadata for that model ID"
    ]


@pytest.mark.parametrize(
    ("unsafe_path", "expected_error"),
    [
        ("workflow-root", "unsafe local Workflow cache directory"),
        ("csrf-file", "unsafe Workflow Builder CSRF file"),
    ],
)
def test_builder_import_rejects_unsafe_cache_symlinks(
    tmp_path,
    unsafe_path,
    expected_error,
):
    cache_root = tmp_path / "cache"
    outside_path = tmp_path / "outside"
    cache_root.mkdir()
    if unsafe_path == "workflow-root":
        outside_path.mkdir()
        (cache_root / "workflow").symlink_to(
            outside_path,
            target_is_directory=True,
        )
    else:
        workflow_local_dir = cache_root / "workflow" / "local"
        workflow_local_dir.mkdir(parents=True)
        outside_path.write_text("attacker-controlled-token")
        (workflow_local_dir / ".csrf").symlink_to(outside_path)

    repository_root = Path(__file__).resolve().parents[6]
    environment = os.environ.copy()
    environment["MODEL_CACHE_DIR"] = str(cache_root)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(repository_root),
            *([existing_python_path] if existing_python_path else []),
        ]
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from inference.core.interfaces.http.builder import routes",
        ],
        cwd=repository_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


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
