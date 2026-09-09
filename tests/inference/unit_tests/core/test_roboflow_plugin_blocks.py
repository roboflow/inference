"""Phase 9: the Roboflow-platform blocks load as a plugin, unchanged - and the
block-disable policy still reaches them."""

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
from inference.roboflow_workflows_plugin import loader as plugin_loader

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


def test_relocated_blocks_now_live_in_the_plugin_package(described_blocks) -> None:
    for block in described_blocks:
        if block.manifest_type_identifier in RELOCATED_BLOCKS:
            assert block.fully_qualified_block_class_name.startswith(
                "inference.roboflow_workflows_plugin."
            )


def test_relocated_blocks_are_contiguous_at_the_head_of_the_plugin_section(
    described_blocks,
) -> None:
    positions = [
        index
        for index, block in enumerate(described_blocks)
        if block.manifest_type_identifier in RELOCATED_BLOCKS
    ]
    assert len(positions) == len(RELOCATED_BLOCKS)
    assert positions == list(range(positions[0], positions[0] + len(positions)))


def test_workflows_core_init_parameters_still_resolve_for_a_relocated_block() -> None:
    """R4 through the real resolution path.

    `retrieve_init_parameters_values` looks up `{block_source}.{param}` first,
    and the core defaults are registered ONLY under `workflows_core.*`
    (`blocks_loader.load_core_blocks_initializers`). A plugin tagged with any
    other source fails to resolve `cache` - Step 14 mutation-checks that.
    """
    from inference.core.workflows.execution_engine.introspection.blocks_loader import (
        load_initializers,
    )
    from inference.core.workflows.execution_engine.v1.compiler.steps_initialiser import (
        retrieve_init_parameters_values,
    )
    from inference.roboflow_workflows_plugin.loader import BLOCKS_SOURCE
    from inference.roboflow_workflows_plugin.sinks.asset_library_attributes.v1 import (
        RoboflowAssetLibraryAttributesBlockV1,
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
        block_source=BLOCKS_SOURCE,
        explicit_init_parameters={
            "workflows_core.cache": shared_cache,
            "workflows_core.api_key": "fake-key",
            "workflows_core.disable_sinks": True,
        },
        initializers=load_initializers(),
    )
    assert values["cache"] is shared_cache
    assert values["disable_sinks"] is True
    assert RoboflowAssetLibraryAttributesBlockV1(**values)._cache is shared_cache


# (types, patterns, identifiers that must survive plugin_loader.load_blocks()) -
# the same shapes core_steps/loader._should_filter_block honours, plus the
# pre-move module path. Patched on the CORE loader module because that is
# where the policy lives and where the plugin reads it from.
POLICY_CASES = [
    (["sink"], [], RELOCATED_BLOCKS - RELOCATED_SINKS),
    ([], ["sinks.roboflow"], RELOCATED_BLOCKS - RELOCATED_SINKS),  # legacy path
    ([], ["integrations.roboflow"], RELOCATED_SINKS),  # legacy path
    ([], ["roboflow_workflows_plugin.sinks"], RELOCATED_BLOCKS - RELOCATED_SINKS),
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
def test_plugin_load_blocks_honours_the_disable_policy(
    types, patterns, expected
) -> None:
    with mock.patch.object(
        core_loader, "WORKFLOW_DISABLED_BLOCK_TYPES", types
    ), mock.patch.object(core_loader, "WORKFLOW_DISABLED_BLOCK_PATTERNS", patterns):
        loaded = {_type_of(block) for block in plugin_loader.load_blocks()}
    assert loaded == expected


NEW_PATH_PROBE = """
import json
import os

from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
)
import inference.core.env as env_module

# CR-1: prove the child actually landed in the requested tensor mode.
requested_tensor_mode = os.environ["ENABLE_TENSOR_DATA_REPRESENTATION"] == "True"
assert env_module.ENABLE_TENSOR_DATA_REPRESENTATION == requested_tensor_mode, (
    env_module.ENABLE_TENSOR_DATA_REPRESENTATION,
    requested_tensor_mode,
)

survivors = {
    b.manifest_type_identifier: b.fully_qualified_block_class_name
    for b in describe_available_blocks(dynamic_blocks=[]).blocks
    if b.fully_qualified_block_class_name.startswith("inference.roboflow_workflows_plugin.")
}

# CR-1: of the two relocated blocks that survive this pattern, one
# (visual_search_classifier) has a _tensor sibling and one (visual_search)
# does not - assert the tensor-mode branch actually selected the right module
# for each, the same way the acceptance test's sink-workflow probe does.
classifier_module = survivors["roboflow_core/visual_search_classifier@v1"]
classifier_leaf = classifier_module.rsplit(".", 1)[0].rsplit(".", 1)[-1]
assert classifier_leaf.endswith("_tensor") == requested_tensor_mode, classifier_module

visual_search_module = survivors["roboflow_core/visual_search@v1"]
visual_search_leaf = visual_search_module.rsplit(".", 1)[0].rsplit(".", 1)[-1]
assert visual_search_leaf == "v1", visual_search_module

print(json.dumps(sorted(survivors)))
"""


@pytest.mark.parametrize("tensor_mode", ["False", "True"])
def test_new_module_path_pattern_disables_the_sinks_in_both_modes(tensor_mode) -> None:
    """The post-move counterpart of the acceptance test's `sinks.roboflow`
    case: a pattern written against the NEW package also works, and the
    plugin's tensor-mode branch selects the right classes in both modes."""
    env = {
        **os.environ,
        "DISABLE_VERSION_CHECK": "True",
        "ENABLE_TENSOR_DATA_REPRESENTATION": tensor_mode,
        # CR-1: pin the effective-mode input; see test_roboflow_sink_acceptance.py.
        "USE_INFERENCE_MODELS": "True",
        "WORKFLOW_DISABLED_BLOCK_PATTERNS": "roboflow_workflows_plugin.sinks",
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT), str(REPO_ROOT / "inference_models")]
        ),
    }
    env.pop("WORKFLOW_DISABLED_BLOCK_TYPES", None)
    result = subprocess.run(
        [sys.executable, "-c", NEW_PATH_PROBE], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout.strip().splitlines()[-1]) == sorted(
        RELOCATED_BLOCKS - RELOCATED_SINKS
    )
