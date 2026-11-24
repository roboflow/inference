"""Guard: V2 suites run without the legacy V1 test bootstrap.

The V1 ``tests/conftest.py`` installs a process configuration at import time
and excludes this directory from its own collection. A V2 run that loads it
would silently test under V1 settings, so this guard stops such a run.
"""

from pathlib import Path

import pytest

from tests.v2.support.paths import WORKFLOWS_ROOT

_LEGACY_CONFTEST = WORKFLOWS_ROOT / "tests" / "conftest.py"
_RIGHT_COMMAND = "cd workflows && python -m pytest -c pytest-v2.ini"


def pytest_configure(config: pytest.Config) -> None:
    loaded_files = {
        Path(plugin.__file__).resolve()
        for plugin in config.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None)
    }
    if _LEGACY_CONFTEST in loaded_files:
        raise pytest.UsageError(
            f"V2 tests must not load the legacy bootstrap {_LEGACY_CONFTEST}. "
            f"Cut conftest lookup at tests/v2: {_RIGHT_COMMAND}"
        )
