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
