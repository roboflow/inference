import dataclasses

import pytest
from filelock import FileLock

from build_scripts.download_fonts import DEFAULT_TARGET_DIR, download_fonts
from inference.core.workflows import configuration as workflows_configuration


@pytest.fixture(scope="session")
def bundled_fonts() -> None:
    """Ensure the approved font assets are available for rendering tests.

    Delegates to the shared downloader used by Docker builds and
    `make download_fonts` - downloads are checksum-verified and skipped when
    assets are already present, so the fixture is a no-op (and network-free)
    on provisioned environments.
    """
    DEFAULT_TARGET_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = DEFAULT_TARGET_DIR / ".download.lock"

    with FileLock(str(lock_path), timeout=300):
        exit_code = download_fonts(DEFAULT_TARGET_DIR, only=[])

    assert exit_code == 0, "Failed to provision approved font assets for tests"


@pytest.fixture
def configured_workflows():
    """Install a WorkflowsConfiguration for one test and restore the previous one.

    Usage:
        def test_x(configured_workflows):
            configured_workflows(tensor={"representation_enabled": True})

    This rebinds only the process registry. Module constants bound in
    `inference.core.workflows.environment` (and in every module that imported
    them) were frozen at import; a test that needs those to change must also
    `importlib.reload` the facade and its consumer - see
    `tests/workflows/unit_tests/core_steps/models/roboflow/action_recognition/test_v1.py`.
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
