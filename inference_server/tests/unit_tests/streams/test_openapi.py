import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"
LEGACY_SNAPSHOT = SCRIPTS / "legacy_openapi_paths.json"

PROBE = """
import json
import sys
from unittest.mock import AsyncMock

from fastapi.testclient import TestClient

from inference_server.app import app
from streamvision.stream_manager.api.entities import (
    CommandContext,
    CommandResponse,
    ListPipelinesResponse,
)

paths = {
    path: sorted(method.upper() for method in operations)
    for path, operations in app.openapi()["paths"].items()
    if path.startswith("/inference_pipelines")
}


class _Client:
    def __init__(self):
        self.list_pipelines = AsyncMock(
            return_value=ListPipelinesResponse(
                status="listed", context=CommandContext(), pipelines=[]
            )
        )
        self.initialise_pipeline = AsyncMock(
            return_value=CommandResponse(status="initialised", context=CommandContext())
        )


app.state.stream_manager_client = _Client()
client = TestClient(app, raise_server_exceptions=False)
listing = client.get("/inference_pipelines/list")
initialised = client.post(
    "/inference_pipelines/initialise",
    json={
        "video_configuration": {"type": "VideoConfiguration", "video_reference": 0},
        "processing_configuration": {
            "type": "WorkflowConfiguration",
            "workspace_name": "ws",
            "workflow_id": "wf",
        },
    },
)


def _status(response):
    try:
        return response.json().get("status")
    except Exception:
        return response.text[:80]


print(
    json.dumps(
        {
            "paths": paths,
            "list": [listing.status_code, _status(listing)],
            "initialise": [initialised.status_code, _status(initialised)],
        }
    )
)
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    directory = tmp_path_factory.mktemp("streams_openapi")
    script = directory / "probe.py"
    script.write_text(PROBE)
    environment = {
        **os.environ,
        "ENABLE_STREAM_API": "True",
        "LEGACY_ROUTES_ENABLED": "True",
        "LEGACY_ROUTE_ENABLED": "True",
        "MODEL_CACHE_DIR": str(directory / "cache"),
    }
    environment.pop("INFERENCE_PRELOAD_MODELS", None)

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.strip().splitlines()[-1])

    return report


def _legacy_stream_paths():
    document = json.loads(LEGACY_SNAPSHOT.read_text())
    paths = {
        path: sorted(methods)
        for path, methods in document["paths"].items()
        if path.startswith("/inference_pipelines")
    }

    return paths


def test_every_legacy_stream_route_is_served_with_the_same_methods(probe):
    legacy_paths = _legacy_stream_paths()

    assert len(legacy_paths) == 8
    assert probe["paths"] == legacy_paths


def test_two_segment_stream_paths_win_over_the_catch_all(probe):
    assert probe["list"] == [200, "listed"]
    assert probe["initialise"] == [200, "initialised"]
