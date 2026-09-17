import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[5]


@pytest.mark.parametrize(
    "module", ["inference.core.env", "inference_models.configuration"]
)
@pytest.mark.parametrize(
    "gateway,expected,warning",
    [
        (
            "gateway.example.test:8443/edge",
            "https://gateway.example.test:8443/edge",
            True,
        ),
        (
            "https://gateway.example.test/edge",
            "https://gateway.example.test/edge",
            False,
        ),
        ("http://127.0.0.1:8080", "http://127.0.0.1:8080", False),
        ("http://gateway.example.test", None, False),
        ("https://user:password@gateway.example.test", None, False),
    ],
)
def test_gateway_configuration_validates_at_import(module, gateway, expected, warning):
    environment = dict(os.environ)
    for key in [
        "SECURE_GATEWAY",
        "LICENSE_SERVER",
        "OFFLINE_MODE",
        "_ROBOFLOW_INFERENCE_OFFLINE_MODE_AT_PROCESS_START",
    ]:
        environment.pop(key, None)
    environment.update(
        SECURE_GATEWAY=gateway,
        DISABLE_VERSION_CHECK="True",
        USE_INFERENCE_MODELS="False",
        PYTHONPATH=os.pathsep.join([str(ROOT), str(ROOT / "inference_models")]),
    )
    code = (
        f"import importlib; print(importlib.import_module({module!r}).SECURE_GATEWAY)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if expected is None:
        assert result.returncode != 0
        assert "ValueError" in result.stderr and "SECURE_GATEWAY" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == expected
        assert ("bare host configuration now uses HTTPS" in result.stderr) is warning
