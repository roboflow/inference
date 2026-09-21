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
