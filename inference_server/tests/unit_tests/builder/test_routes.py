import json
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


def test_builder_html_injects_csrf(builder_app):
    client = TestClient(builder_app)
    response = client.get("/build")
    assert response.status_code == HTTP_200_OK
    assert "text/html" in response.headers["content-type"]

    token_match = re.search(r"CSRF=([0-9a-f]+)", response.text)
    assert token_match, "CSRF token not found in HTML response"
    token = token_match.group(1)
    assert len(token) == 32, "CSRF token should be 32 hex digits long"


def test_builder_edit_injects_csrf(builder_app):
    client = TestClient(builder_app)
    response = client.get("/build/edit/my-workflow")
    assert response.status_code == HTTP_200_OK
    token_match = re.search(r"CSRF=([0-9a-f]+)", response.text)
    assert token_match, "CSRF token not found in HTML response"
    token = token_match.group(1)
    assert len(token) == 32, "CSRF token should be 32 hex digits long"


def test_builder_redirect_trailing_slash(builder_app):
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


def test_api_get_workflow_invalid_id(builder_app, builder_env):
    client = TestClient(builder_app)

    invalid_id = "invalid$id"
    response = client.get(
        f"/build/api/{invalid_id}",
        headers={"X-CSRF": builder_env.csrf},
    )
    assert response.status_code == HTTP_400_BAD_REQUEST


def test_api_create_and_read(builder_app, builder_env):
    client = TestClient(builder_app)

    create_resp = client.post(
        "/build/api/test-wf",
        json={"id": "test-wf", "stuff": 123},
        headers={"X-CSRF": builder_env.csrf},
    )
    assert create_resp.status_code == HTTP_201_CREATED

    get_resp = client.get("/build/api/test-wf", headers={"X-CSRF": builder_env.csrf})
    assert get_resp.status_code == HTTP_200_OK
    data = get_resp.json()
    assert data["data"]["config"] == {"id": "test-wf", "stuff": 123}


def test_api_create_rejects_final_workflow_symlink(builder_app, builder_env, tmp_path):
    routes = builder_env

    workflow_id = "write-symlink"
    workflow_file, outside_file, outside_contents = _install_final_workflow_symlink(
        routes=routes,
        tmp_path=tmp_path,
        workflow_id=workflow_id,
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


def test_api_get_rejects_final_workflow_symlink(builder_app, builder_env, tmp_path):
    routes = builder_env

    workflow_id = "read-symlink"
    workflow_file, outside_file, outside_contents = _install_final_workflow_symlink(
        routes=routes,
        tmp_path=tmp_path,
        workflow_id=workflow_id,
    )
    client = TestClient(builder_app)

    response = client.get(
        f"/build/api/{workflow_id}",
        headers={"X-CSRF": routes.csrf},
    )

    assert response.status_code == HTTP_404_NOT_FOUND
    assert workflow_file.is_symlink()
    assert json.loads(outside_file.read_text()) == outside_contents


def test_api_delete_rejects_final_workflow_symlink(builder_app, builder_env, tmp_path):
    routes = builder_env

    workflow_id = "delete-symlink"
    workflow_file, outside_file, outside_contents = _install_final_workflow_symlink(
        routes=routes,
        tmp_path=tmp_path,
        workflow_id=workflow_id,
    )
    client = TestClient(builder_app)

    response = client.delete(
        f"/build/api/{workflow_id}",
        headers={"X-CSRF": routes.csrf},
    )

    assert response.status_code == HTTP_404_NOT_FOUND
    assert workflow_file.is_symlink()
    assert json.loads(outside_file.read_text()) == outside_contents


def test_api_list_skips_final_workflow_symlink(builder_app, builder_env, tmp_path):
    routes = builder_env

    workflow_id = "list-symlink"
    workflow_file, outside_file, outside_contents = _install_final_workflow_symlink(
        routes=routes,
        tmp_path=tmp_path,
        workflow_id=workflow_id,
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
    builder_env,
    monkeypatch,
):
    routes = builder_env

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

    package_root = Path(__file__).resolve().parents[3]
    environment = os.environ.copy()
    environment["MODEL_CACHE_DIR"] = str(cache_root)
    existing_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = os.pathsep.join(
        [
            str(package_root),
            *([existing_python_path] if existing_python_path else []),
        ]
    )

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from inference_server.builder import routes",
        ],
        cwd=package_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr


def test_fallback_redirect_invalid_id(builder_app):
    client = TestClient(builder_app)
    response = client.get("/build/../../etc/passwd", follow_redirects=False)
    assert (
        response.status_code == HTTP_404_NOT_FOUND
    ), f"Expected 404, got {response.status_code}"


def test_fallback_redirect_exists(builder_app, builder_env):
    client = TestClient(builder_app)

    client.post(
        "/build/api/foobar",
        json={"id": "foobar"},
        headers={"X-CSRF": builder_env.csrf},
    )
    response = client.get("/build/foobar", follow_redirects=False)
    assert response.status_code == HTTP_302_FOUND
    assert response.headers["location"] == "/build/edit/foobar"


def test_fallback_redirect_not_exists(builder_app):
    client = TestClient(builder_app)
    response = client.get("/build/does-not-exist", follow_redirects=False)
    assert response.status_code == HTTP_302_FOUND
    assert response.headers["location"] == "/build"


def test_builder_route_absent_when_router_not_mounted():
    app = FastAPI()

    client = TestClient(app)
    get_response = client.get("/build")
    head_response = client.head("/build")

    assert get_response.status_code == 404
    assert head_response.status_code == 404


def test_get_probes_builder_availability_when_mounted(builder_app):
    client = TestClient(builder_app)

    response = client.get("/build")

    assert response.status_code == HTTP_200_OK
    assert "text/html" in response.headers["content-type"]


def test_head_does_not_reliably_report_builder_availability(builder_app):
    client = TestClient(builder_app)

    response = client.head("/build")

    assert response.status_code != HTTP_200_OK
