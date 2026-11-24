"""Guard: Streamvision V2 suites run without the default test tree's conftests."""

from pathlib import Path

import pytest

_DEFAULT_CONFTEST = Path(__file__).resolve().parent.parent / "conftest.py"
_RIGHT_COMMAND = "cd stream_vision && python -m pytest -c pytest-v2.ini"


def pytest_configure(config: pytest.Config) -> None:
    loaded_files = {
        Path(plugin.__file__).resolve()
        for plugin in config.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None)
    }
    if _DEFAULT_CONFTEST in loaded_files:
        raise pytest.UsageError(
            f"V2 tests must not load {_DEFAULT_CONFTEST}. "
            f"Cut conftest lookup at tests/v2: {_RIGHT_COMMAND}"
        )
