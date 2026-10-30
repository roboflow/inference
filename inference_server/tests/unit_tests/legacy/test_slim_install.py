import json
import os
import subprocess
import sys


def test_legacy_package_imports_without_roboflow_workflows():
    code = (
        "import sys; sys.modules['roboflow_workflows'] = None; "
        "import inference_server.legacy.entities, inference_server.legacy.common, "
        "inference_server.legacy.errors, inference_server.legacy.bridge; print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_app_imports_and_drops_workflow_routes_without_roboflow_workflows():
    code = (
        "import importlib.util; _find = importlib.util.find_spec; "
        "importlib.util.find_spec = lambda name, *a, **k: "
        "None if name == 'roboflow_workflows' else _find(name, *a, **k); "
        "import inference_server.app as app_mod; "
        "paths = set(app_mod.app.openapi()['paths']); "
        "assert '/workflows/run' not in paths, sorted(paths); print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_app_import_fails_when_roboflow_workflows_is_broken():
    # Claim the package is installed, then break it: a broken install must crash
    # the app, never degrade to the routes-dropped path. Faking find_spec keeps
    # this meaningful on slim installs, where the package really is absent.
    code = (
        "import importlib.util, sys; _find = importlib.util.find_spec; "
        "importlib.util.find_spec = lambda name, *a, **k: "
        "object() if name == 'roboflow_workflows' else _find(name, *a, **k); "
        "sys.modules['roboflow_workflows.http_contract'] = None; "
        "import inference_server.app"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0, result.stdout
    assert "roboflow_workflows" in result.stderr, result.stderr


def test_app_imports_with_builder_enabled_without_roboflow_workflows(tmp_path):
    code = (
        "import asyncio, importlib.util, sys; _find = importlib.util.find_spec; "
        "importlib.util.find_spec = lambda name, *a, **k: "
        "None if name == 'roboflow_workflows' else _find(name, *a, **k); "
        "sys.modules['roboflow_workflows'] = None; "
        "import inference_server.app as app_mod; "
        "paths = set(app_mod.app.openapi()['paths']); "
        "assert '/build' in paths, sorted(paths); "
        "from inference_server.builder import models; "
        "bridge = type('B', (), "
        "{'describe': lambda self: asyncio.sleep(0, result=[])})(); "
        "listed = asyncio.run(models.list_models(bridge)); "
        "assert not any(m['is_foundation'] for m in listed), listed; "
        "assert all(m['compatible_block_types'] == [] for m in listed), listed; "
        "print('ok')"
    )
    environment = os.environ.copy()
    environment["ENABLE_BUILDER"] = "true"
    environment["MODEL_CACHE_DIR"] = str(tmp_path / "cache")
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=environment,
    )
    assert result.returncode == 0, result.stderr


def test_startup_without_the_workflows_package_serves_inference():
    code = """
import base64, io, json, sys
sys.modules["roboflow_workflows"] = None
sys.path.insert(0, ".")
from types import SimpleNamespace
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
import inference_model_manager.watchdogs as watchdogs
import inference_server.gateway_resolver as resolver
from inference_server import configuration
from inference_server.framework import model_stat
from tests.unit_tests.legacy.conftest import FakeGateway

watchdogs.start_enabled_watchdogs = lambda: []
configuration.ACTIVE_LEARNING_ENABLED = True
model_stat.get_one_page_of_model_metadata = lambda model_id, api_key=None, **_: (
    SimpleNamespace(task_type="object-detection")
)
detection = SimpleNamespace(
    xyxy=np.array([[1, 1, 3, 5]], dtype=float),
    confidence=np.array([0.9]),
    class_id=np.array([0]),
)
gateway = FakeGateway(
    predictions={("ds/1", "infer"): detection},
    model_info={"ds/1": {"class_names": ["cat"], "actions": {"infer": {}}}},
)
resolver.resolve_gateway = lambda: gateway
import inference_server.app as app_mod
from inference_server.legacy import active_learning_registration as registration

app_mod._start_usage_collector = lambda: None
buffer = io.BytesIO()
Image.new("RGB", (8, 6)).save(buffer, format="PNG")
body = {
    "model_id": "ds/1",
    "api_key": "k",
    "image": {"type": "base64", "value": base64.b64encode(buffer.getvalue()).decode()},
}
with TestClient(app_mod.app, raise_server_exceptions=False) as client:
    response = client.post("/infer/object_detection", json=body)
    registrar = registration._REGISTRAR
print(json.dumps({
    "status": response.status_code,
    "registrar_absent": registrar is None,
    "loaded": sorted(
        name for name, module in sys.modules.items()
        if name.startswith("roboflow_workflows") and module is not None
    ),
}))
"""
    root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=dict(os.environ),
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr
    outcome = json.loads(result.stdout.strip().splitlines()[-1])
    assert outcome == {"status": 200, "registrar_absent": True, "loaded": []}
