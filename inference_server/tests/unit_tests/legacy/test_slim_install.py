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
        "paths = {r.path for r in app_mod.app.routes if hasattr(r, 'path')}; "
        "assert '/workflows/run' not in paths, sorted(paths); print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_app_import_fails_when_roboflow_workflows_is_broken():
    code = (
        "import sys; sys.modules['roboflow_workflows.http_contract'] = None; "
        "import inference_server.app"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode != 0, result.stdout
    assert "roboflow_workflows.http_contract" in result.stderr, result.stderr
