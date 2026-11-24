"""Bootstrap for the V1/V2 compatibility suite: production V1 tensor settings.

These tests compare V2 with V1 production code, so they run with the V1 tensor
configuration taken from ``ENABLE_TENSOR_DATA_REPRESENTATION``, exactly as the
legacy ``tests/conftest.py`` installs it, but without loading that conftest or
its fixtures. Run both modes::

    cd workflows
    ENABLE_TENSOR_DATA_REPRESENTATION=False python -m pytest -c pytest.ini \
        --confcutdir=tests/compatibility/v2 tests/compatibility/v2
    ENABLE_TENSOR_DATA_REPRESENTATION=True python -m pytest -c pytest.ini \
        --confcutdir=tests/compatibility/v2 tests/compatibility/v2
"""

import dataclasses
import os
from pathlib import Path

import pytest
from roboflow_workflows import configuration

from tests.v2.support.paths import WORKFLOWS_ROOT

_LEGACY_CONFTEST = WORKFLOWS_ROOT / "tests" / "conftest.py"
_RIGHT_COMMAND = (
    "cd workflows && python -m pytest -c pytest.ini "
    "--confcutdir=tests/compatibility/v2 tests/compatibility/v2"
)

TENSOR_MODE = (
    os.environ.get("ENABLE_TENSOR_DATA_REPRESENTATION", "False").lower() == "true"
)

# Installed at import, before any test module imports V1 code, as the legacy
# bootstrap does. Duplicated on purpose: the legacy conftest stays unchanged.
_base_configuration = configuration.default_configuration()
configuration.reset_configuration()
configuration.configure_process(
    dataclasses.replace(
        _base_configuration,
        tensor=dataclasses.replace(
            _base_configuration.tensor,
            representation_enabled=TENSOR_MODE,
            image_tensor_device=configuration.resolve_image_tensor_device(
                TENSOR_MODE
            ),
        ),
    )
)


def pytest_configure(config: pytest.Config) -> None:
    loaded_files = {
        Path(plugin.__file__).resolve()
        for plugin in config.pluginmanager.get_plugins()
        if getattr(plugin, "__file__", None)
    }
    if _LEGACY_CONFTEST in loaded_files:
        raise pytest.UsageError(
            f"Compatibility tests must not load the legacy bootstrap "
            f"{_LEGACY_CONFTEST}. Cut conftest lookup: {_RIGHT_COMMAND}"
        )
