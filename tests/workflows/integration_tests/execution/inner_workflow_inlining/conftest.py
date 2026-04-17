"""Register stub workflow blocks used by ``echo_child_workflow`` in ``_common``."""

from unittest import mock

import pytest

from inference.core.workflows.execution_engine.introspection import blocks_loader

_SCALAR_ONLY_ECHO_PLUGIN = (
    "tests.workflows.integration_tests.execution.stub_plugins.scalar_only_block_plugin"
)


@pytest.fixture(autouse=True)
def _scalar_only_echo_plugin():
    with mock.patch.object(
        blocks_loader,
        "get_plugin_modules",
        return_value=[_SCALAR_ONLY_ECHO_PLUGIN],
    ):
        blocks_loader.clear_caches()
        yield
    blocks_loader.clear_caches()
