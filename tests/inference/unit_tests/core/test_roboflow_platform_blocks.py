"""The Roboflow-platform blocks are core blocks: listed with the core source,
reached through the injected platform client, and subject to the core
block-disable policy."""

import json
import os
import pathlib
import subprocess
import sys
from unittest import mock

import pytest

from inference.core.workflows.core_steps import loader as core_loader
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

# Read off the blocks' `type` Literals in the tree - the public contract.
RELOCATED_BLOCKS = {
    "roboflow_core/roboflow_dataset_upload@v1",
    "roboflow_core/roboflow_dataset_upload@v2",
    "roboflow_core/roboflow_custom_metadata@v1",
    "roboflow_core/model_monitoring_inference_aggregator@v1",
    "roboflow_core/asset_library_attributes@v1",
    "roboflow_core/roboflow_vision_events@v1",
    "roboflow_core/vision_event_bundle@v1",
    "roboflow_core/visual_search@v1",
    "roboflow_core/visual_search_classifier@v1",
}
RELOCATED_SINKS = RELOCATED_BLOCKS - {
    "roboflow_core/visual_search@v1",
    "roboflow_core/visual_search_classifier@v1",
}


def _type_of(block_class) -> str:
    return block_class.get_manifest().model_fields["type"].annotation.__args__[0]


@pytest.fixture
def described_blocks():
    return describe_available_blocks(dynamic_blocks=[]).blocks


def test_every_relocated_block_is_still_available(described_blocks) -> None:
    available = {b.manifest_type_identifier for b in described_blocks}
    missing = RELOCATED_BLOCKS - available
    assert not missing, f"relocated blocks disappeared: {missing}"


def test_relocated_blocks_keep_the_core_block_source(described_blocks) -> None:
    for block in described_blocks:
        if block.manifest_type_identifier in RELOCATED_BLOCKS:
            assert block.block_source == "workflows_core", (
                f"{block.manifest_type_identifier} reports "
                f"block_source={block.block_source!r}"
            )


def test_relocated_blocks_report_their_historic_core_class_path(
    described_blocks,
) -> None:
    for block in described_blocks:
        if block.manifest_type_identifier in RELOCATED_BLOCKS:
            assert block.fully_qualified_block_class_name.startswith(
                (
                    "inference.core.workflows.core_steps.sinks.roboflow.",
                    "inference.core.workflows.core_steps.integrations.roboflow.",
                )
            ), block.fully_qualified_block_class_name


def test_every_relocated_block_takes_the_platform_client() -> None:
    blocks = {
        _type_of(block): block
        for block in core_loader.load_blocks()
        if _type_of(block) in RELOCATED_BLOCKS
    }
    # The bundle sink writes to disk and never calls the Roboflow API.
    calling_blocks = RELOCATED_BLOCKS - {"roboflow_core/vision_event_bundle@v1"}
    assert set(blocks) == RELOCATED_BLOCKS
    for block_type in calling_blocks:
        assert "platform_client" in blocks[block_type].get_init_parameters(), block_type


def test_workflows_core_init_parameters_resolve_for_a_relocated_block() -> None:
    """Through the real resolution path: `retrieve_init_parameters_values`
    looks up `{block_source}.{param}`, and the server binds its platform
    client as `workflows_core.platform_client`."""
    from roboflow_workflows.core_steps.sinks.roboflow.asset_library_attributes.v1 import (
        RoboflowAssetLibraryAttributesBlockV1,
    )

    from inference.core.interfaces.roboflow_platform_client import (
        SERVER_PLATFORM_CLIENT,
        install_workflows_platform_bindings,
    )
    from inference.core.workflows.execution_engine.introspection.blocks_loader import (
        load_initializers,
    )
    from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
        retrieve_init_parameters_values,
    )

    class _Cache:
        def get(self, key):
            return None

        def set(self, key, value, expire=None):
            return None

    shared_cache = _Cache()
    values = retrieve_init_parameters_values(
        block_name="attributes",
        block_init_parameters=RoboflowAssetLibraryAttributesBlockV1.get_init_parameters(),
        block_source="workflows_core",
        explicit_init_parameters=install_workflows_platform_bindings(
            {
                "workflows_core.cache": shared_cache,
                "workflows_core.api_key": "fake-key",
                "workflows_core.disable_sinks": True,
            }
        ),
        initializers=load_initializers(),
    )
    assert values["cache"] is shared_cache
    assert values["disable_sinks"] is True
    assert values["platform_client"] is SERVER_PLATFORM_CLIENT
    block = RoboflowAssetLibraryAttributesBlockV1(**values)
    assert block._cache is shared_cache
    assert block._platform_client is SERVER_PLATFORM_CLIENT


# (types, patterns, relocated identifiers that must survive core load_blocks()) -
# the shapes core_steps/loader._should_filter_block honours, against both the
# canonical and the historic module path.
POLICY_CASES = [
    (["sink"], [], RELOCATED_BLOCKS - RELOCATED_SINKS),
    ([], ["sinks.roboflow"], RELOCATED_BLOCKS - RELOCATED_SINKS),  # legacy path
    ([], ["integrations.roboflow"], RELOCATED_SINKS),  # legacy path
    ([], ["core_steps.sinks.roboflow"], RELOCATED_BLOCKS - RELOCATED_SINKS),
    (
        [],
        ["roboflowdatasetuploadblockv1"],
        RELOCATED_BLOCKS - {"roboflow_core/roboflow_dataset_upload@v1"},
    ),
    # NOTE: "visual search" is also a substring of visual_search_classifier's
    # display name ("Roboflow Visual Search Classifier"), so both blocks are
    # filtered - verified directly against core_loader._should_filter_block.
    (
        [],
        ["visual search"],
        RELOCATED_BLOCKS
        - {
            "roboflow_core/visual_search@v1",
            "roboflow_core/visual_search_classifier@v1",
        },
    ),
    ([], ["nonexistent_pattern"], RELOCATED_BLOCKS),
    ([], [], RELOCATED_BLOCKS),
]


@pytest.mark.parametrize("types,patterns,expected", POLICY_CASES)
def test_core_load_blocks_honours_the_disable_policy(types, patterns, expected) -> None:
    with mock.patch.object(
        core_loader, "WORKFLOW_DISABLED_BLOCK_TYPES", types
    ), mock.patch.object(core_loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", patterns):
        loaded = {_type_of(block) for block in core_loader.load_blocks()}
    assert loaded & RELOCATED_BLOCKS == expected


# The 2 identifiers that survive `WORKFLOW_DISABLED_BLOCK_PATTERNS=
# "roboflow_workflows.core_steps.sinks.roboflow"`: visual_search@v1 has no _tensor sibling
# (its module leaf is always "v1"); visual_search_classifier@v1 does.
_SURVIVOR_EXPECTED_LEAF = {
    "roboflow_core/visual_search@v1": {"False": "v1", "True": "v1"},
    "roboflow_core/visual_search_classifier@v1": {"False": "v1", "True": "v1_tensor"},
}

NEW_PATH_PROBE = """
import json
import os

from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
)
import inference.core.env as env_module

RELOCATED_BLOCKS = %r
survivors = {
    b.manifest_type_identifier: b.fully_qualified_block_class_name
    for b in describe_available_blocks(dynamic_blocks=[]).blocks
    if b.manifest_type_identifier in RELOCATED_BLOCKS
}
# CR-1: report - do not just self-assert - the effective tensor flag and the
# exact module leaf selected for each surviving identifier, so the PARENT can
# independently verify them (mirrors test_roboflow_sink_acceptance.py).
module_leaf_by_identifier = {
    identifier: module_name.rsplit(".", 1)[0].rsplit(".", 1)[-1]
    for identifier, module_name in survivors.items()
}

print(
    json.dumps(
        {
            "identifiers": sorted(survivors),
            "effective_tensor_mode": env_module.ENABLE_TENSOR_DATA_REPRESENTATION,
            "module_leaf_by_identifier": module_leaf_by_identifier,
        }
    )
)
"""


@pytest.mark.parametrize("tensor_mode", ["False", "True"])
def test_new_module_path_pattern_disables_the_sinks_in_both_modes(tensor_mode) -> None:
    """A pattern written against the canonical package also works, and the
    core loader's tensor-mode branch selects the right classes in both modes."""
    env = {
        **os.environ,
        "DISABLE_VERSION_CHECK": "True",
        "ENABLE_TENSOR_DATA_REPRESENTATION": tensor_mode,
        # CR-1: pin the effective-mode input; see test_roboflow_sink_acceptance.py.
        "USE_INFERENCE_MODELS": "True",
        "WORKFLOW_DISABLED_BLOCK_PATTERNS": "roboflow_workflows.core_steps.sinks.roboflow",
        "PYTHONPATH": os.pathsep.join(
            [
                str(REPO_ROOT),
                str(REPO_ROOT / "inference_models"),
                str(REPO_ROOT / "stream_vision"),
            ]
        ),
    }
    env.pop("WORKFLOW_DISABLED_BLOCK_TYPES", None)
    result = subprocess.run(
        [sys.executable, "-c", NEW_PATH_PROBE % (sorted(RELOCATED_BLOCKS),)],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    survivor_identifiers = sorted(RELOCATED_BLOCKS - RELOCATED_SINKS)
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {
        "identifiers": survivor_identifiers,
        "effective_tensor_mode": tensor_mode == "True",
        "module_leaf_by_identifier": {
            identifier: _SURVIVOR_EXPECTED_LEAF[identifier][tensor_mode]
            for identifier in survivor_identifiers
        },
    }
