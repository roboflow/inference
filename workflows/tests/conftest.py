"""Package-owned fixtures. No `inference` server imports allowed here.

Configuration is installed via `roboflow_workflows.configuration` directly.
Font assets are provisioned via the sibling `workflows/build_scripts/download_fonts.py`
(owned by the packaging worker).
"""

import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import pytest
from filelock import FileLock
from roboflow_workflows import configuration as workflows_configuration

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_FONTS_HELPER = _PROJECT_ROOT / "build_scripts" / "download_fonts.py"
_FONTS_ASSETS_DIR = (
    _PROJECT_ROOT
    / "roboflow_workflows"
    / "core_steps"
    / "visualizations"
    / "common"
    / "fonts"
    / "assets"
)

_TENSOR_MODE = (
    os.environ.get("ENABLE_TENSOR_DATA_REPRESENTATION", "False").lower() == "true"
)

# Freeze WorkflowsConfiguration BEFORE any test module import runs. Package
# tests must never depend on inference.core.env; the standalone bootstrap
# resolves tensor mode from ENABLE_TENSOR_DATA_REPRESENTATION (existing CI
# knob) and installs it explicitly here so consumers that read
# environment.py / core_steps/loader.py at import time see the right value.
_base_configuration = workflows_configuration.default_configuration()
workflows_configuration.reset_configuration()
workflows_configuration.configure_process(
    dataclasses.replace(
        _base_configuration,
        tensor=dataclasses.replace(
            _base_configuration.tensor,
            representation_enabled=_TENSOR_MODE,
            image_tensor_device=workflows_configuration.resolve_image_tensor_device(
                _TENSOR_MODE
            ),
        ),
    )
)


@pytest.fixture(scope="session")
def bundled_fonts() -> None:
    """Provision approved font assets via the package's own downloader.

    Never reaches into the repository's root `build_scripts/`; the helper here
    is the standalone project's own copy. Checksum-verified, network-free when
    already provisioned.
    """
    _FONTS_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = _FONTS_ASSETS_DIR / ".download.lock"
    with FileLock(str(lock_path), timeout=300):
        result = subprocess.run(
            [sys.executable, str(_FONTS_HELPER)],
            capture_output=True,
            text=True,
        )
    assert (
        result.returncode == 0
    ), f"font provisioning failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"


@pytest.fixture
def configured_workflows():
    """Install a WorkflowsConfiguration for one test and restore the previous.

    Sanctioned per-test entry point when a case needs to flip a knob for one
    test. Module-level default (tensor on/off) was already frozen at import
    time from ENABLE_TENSOR_DATA_REPRESENTATION.
    """
    previous = workflows_configuration._CONFIGURATION

    def _install(**group_overrides):
        base = workflows_configuration.default_configuration()
        groups = {
            name: dataclasses.replace(getattr(base, name), **overrides)
            for name, overrides in group_overrides.items()
        }
        configuration = dataclasses.replace(base, **groups)
        workflows_configuration.reset_configuration()
        workflows_configuration.configure_process(configuration)
        return configuration

    yield _install
    with workflows_configuration._INSTALL_LOCK:
        workflows_configuration._CONFIGURATION = previous
