import json
import os
import subprocess
import sys

from inference.core.env import ENTERPRISE_BLOCKS_PLUGIN as ENTERPRISE_PLUGIN
from inference.core.env import ROBOFLOW_BLOCKS_PLUGIN as ROBOFLOW_PLUGIN

MQTT_BLOCK = (
    "inference.enterprise.workflows.enterprise_blocks.sinks.mqtt_writer.v1"
    ".MQTTWriterSinkBlockV1"
)
FAKE_PLUGIN = "fake_workflows_plugin"
FAKE_BLOCK = "__main__.FakePluginBlockV1"

# Installs `FAKE_PLUGIN` in sys.modules before anything imports the loader, so
# `WORKFLOWS_PLUGINS=fake_workflows_plugin` resolves without a real distribution.
_FAKE_PLUGIN_SETUP = """
import sys
import types
from typing import List, Literal, Type

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)


class FakePluginManifest(WorkflowBlockManifest):
    type: Literal["fake_plugin_block@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="value")]


class FakePluginBlockV1(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return FakePluginManifest

    def run(self) -> BlockResult:
        return {"value": 1}


_module = types.ModuleType("fake_workflows_plugin")
_module.load_blocks = lambda: [FakePluginBlockV1]
sys.modules["fake_workflows_plugin"] = _module
"""

_LOAD_AND_REPORT = """
import json
import os

from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    load_workflow_blocks,
)

print(
    json.dumps(
        {
            "plugins": os.environ.get("WORKFLOWS_PLUGINS"),
            "blocks": [[b.identifier, b.block_source] for b in load_workflow_blocks()],
        }
    )
)
"""

PROBE = _LOAD_AND_REPORT
ORDERING_PROBE = _FAKE_PLUGIN_SETUP + _LOAD_AND_REPORT


def _run_probe(probe: str, **env: str) -> dict:
    child_env = {**os.environ, "DISABLE_VERSION_CHECK": "True", **env}
    # Never inherit the parent's plugin list: env.py may already have expanded
    # the enterprise plugin into it, which would mask the "disabled" case.
    child_env.pop("WORKFLOWS_PLUGINS", None)
    child_env.update(env)  # ... unless the case sets it explicitly
    result = subprocess.run(
        [sys.executable, "-c", probe], env=child_env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _block_sources(**env: str) -> dict:
    return dict(_run_probe(PROBE, **env)["blocks"])


def test_enterprise_blocks_load_under_the_core_source_when_enabled() -> None:
    sources = _block_sources(LOAD_ENTERPRISE_BLOCKS="True")
    assert sources[MQTT_BLOCK] == "workflows_core"


def test_enterprise_blocks_are_absent_when_disabled() -> None:
    sources = _block_sources(LOAD_ENTERPRISE_BLOCKS="False")
    assert MQTT_BLOCK not in sources


def test_enterprise_blocks_keep_their_position_between_core_and_custom_plugins() -> (
    None
):
    # Before this refactor enterprise blocks were merged into the core list, so
    # the order was core -> enterprise -> custom plugins. Appending the plugin
    # to WORKFLOWS_PLUGINS would silently flip that to core -> custom ->
    # enterprise for anyone who also configures their own plugin.
    report = _run_probe(
        ORDERING_PROBE,
        LOAD_ENTERPRISE_BLOCKS="True",
        WORKFLOWS_PLUGINS=FAKE_PLUGIN,
    )
    identifiers = [identifier for identifier, _ in report["blocks"]]
    sources = dict(report["blocks"])

    assert sources[MQTT_BLOCK] == "workflows_core"
    assert sources[FAKE_BLOCK] == FAKE_PLUGIN
    # Enterprise blocks are tagged `workflows_core` too (that is the whole
    # point of BLOCKS_SOURCE), so the genuine core blocks have to be picked
    # out by module path - otherwise MQTT -> core -> fake also satisfies the
    # assertions and the ordering regression slips through.
    genuine_core_positions = [
        index
        for index, (identifier, source) in enumerate(report["blocks"])
        if source == "workflows_core"
        and not identifier.startswith("inference.enterprise.")
        and not identifier.startswith("inference.roboflow_workflows_plugin.")
    ]
    mqtt_position = identifiers.index(MQTT_BLOCK)
    fake_position = identifiers.index(FAKE_BLOCK)

    # every genuine core block precedes the enterprise block ...
    assert max(genuine_core_positions) < mqtt_position
    # ... which in turn precedes the custom plugin's block.
    assert mqtt_position < fake_position
    # The Roboflow-platform plugin is expanded unconditionally and prepended
    # ahead of the enterprise one (env.py), so the historical core -> roboflow
    # -> enterprise -> custom order is preserved.
    assert report["plugins"] == f"{ROBOFLOW_PLUGIN},{ENTERPRISE_PLUGIN},{FAKE_PLUGIN}"


def test_env_expansion_does_not_duplicate_an_already_listed_enterprise_plugin() -> None:
    # A duplicated plugin name loads the same blocks twice and trips the
    # manifest-identifier clash check in describe_available_blocks.
    report = _run_probe(
        PROBE,
        LOAD_ENTERPRISE_BLOCKS="True",
        WORKFLOWS_PLUGINS=ENTERPRISE_PLUGIN,
    )
    assert report["plugins"] == f"{ROBOFLOW_PLUGIN},{ENTERPRISE_PLUGIN}"
    identifiers = [identifier for identifier, _ in report["blocks"]]
    assert identifiers.count(MQTT_BLOCK) == 1


def test_env_expansion_does_not_duplicate_an_already_listed_roboflow_plugin() -> None:
    report = _run_probe(PROBE, WORKFLOWS_PLUGINS=ROBOFLOW_PLUGIN)
    assert report["plugins"] == ROBOFLOW_PLUGIN
    identifiers = [identifier for identifier, _ in report["blocks"]]
    roboflow_blocks = [
        identifier
        for identifier in identifiers
        if identifier.startswith("inference.roboflow_workflows_plugin.")
    ]
    assert len(roboflow_blocks) == len(set(roboflow_blocks)) == 9
