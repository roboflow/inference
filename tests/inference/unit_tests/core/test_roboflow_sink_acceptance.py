"""Acceptance test for Phase 9's relocation (spec-phase9.md:22) and for the
block-disable policy the relocation must not bypass (round-3 defect 1).

Part 1 compiles and runs a real Roboflow-sink workflow with
`workflows_core.disable_sinks=True` and an injected shared cache, in BOTH
tensor modes, and compares against a recorded baseline. The same file runs
before and after the move: the blocks' `type` identifiers do not change.

Part 2 records which relocated block identifiers survive discovery under
`WORKFLOW_DISABLED_BLOCK_TYPES` / `WORKFLOW_DISABLED_BLOCK_PATTERNS`, and
whether a workflow naming a disabled one still compiles. One case uses the OLD
module path (`sinks.roboflow`): an operator's existing pattern has to keep
working after the move.

The workflow is the one the existing network-free integration test already
compiles and runs
(`tests/workflows/integration_tests/execution/test_workflow_with_asset_library_attributes.py:12-36`),
so the "before" side is real, not invented.

Every case needs a subprocess: `ENABLE_TENSOR_DATA_REPRESENTATION` and the two
policy variables are read at import time by `core_steps/loader.py` (and, after
the move, by the plugin loader). The child's PYTHONPATH must carry the REPO
ROOT as well as `inference_models` - with `inference_models` alone the venv's
editable .pth resolves `inference` from a different checkout and the child dies
on `ModuleNotFoundError: inference_models.utils.performance`.

CR-1: the effective tensor mode is `ENABLE_TENSOR_DATA_REPRESENTATION AND
USE_INFERENCE_MODELS` (inference/core/env.py:1486), so a subprocess builder
that sets only the former can silently run NumPy in both "modes" if it
inherits `USE_INFERENCE_MODELS=False` from the shell. `_run()` below pins
`USE_INFERENCE_MODELS=True` and every child asserts, from inside the child,
that `inference.core.env.ENABLE_TENSOR_DATA_REPRESENTATION` equals the
requested mode. The sink-workflow probe additionally asserts which module was
actually selected for the 7 blocks with a `_tensor` sibling, and that the 2
blocks with no sibling resolve to the same module in both modes.
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[4]

WORKFLOW = {
    "version": "1.3.0",
    "inputs": [
        {"type": "WorkflowBatchInput", "name": "source_id", "kind": ["string"]},
        {"type": "WorkflowParameter", "name": "location"},
        {"type": "WorkflowParameter", "name": "extra_tag"},
    ],
    "steps": [
        {
            "type": "roboflow_core/asset_library_attributes@v1",
            "name": "asset_library_attributes",
            "source_id": "$inputs.source_id",
            "metadata": {"location": "$inputs.location"},
            "tags": ["$inputs.extra_tag"],
            "disable_sink": False,
        }
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "message",
            "selector": "$steps.asset_library_attributes.message",
        }
    ],
}

PROBE = """
import json
import os

from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference.core.workflows.execution_engine.introspection.blocks_loader import (
    describe_available_blocks,
)
import inference.core.env as env_module

# CR-1: prove the child actually landed in the requested tensor mode, not
# just that it read the env var back unmodified.
requested_tensor_mode = os.environ["ENABLE_TENSOR_DATA_REPRESENTATION"] == "True"
assert env_module.ENABLE_TENSOR_DATA_REPRESENTATION == requested_tensor_mode, (
    env_module.ENABLE_TENSOR_DATA_REPRESENTATION,
    requested_tensor_mode,
)

WORKFLOW = json.loads(%s)


class SharedCache:
    def __init__(self):
        self.storage = {}

    def get(self, key):
        return self.storage.get(key)

    def set(self, key, value, expire=None):
        self.storage[key] = value


cache = SharedCache()
engine = ExecutionEngine.init(
    workflow_definition=WORKFLOW,
    init_parameters={
        "workflows_core.model_manager": None,
        "workflows_core.api_key": "my_api_key",
        "workflows_core.cache": cache,
        "workflows_core.update_attributes_offloader": None,
        "workflows_core.disable_sinks": True,
    },
)
# v1/core.py:367 stores `_compiled_workflow`; CompiledWorkflow.steps holds the
# InitialisedStep records (compiler/entities.py:60).
step = engine._engine._compiled_workflow.steps["asset_library_attributes"].step
result = engine.run(
    runtime_parameters={
        "source_id": ["img-1", "img-2"],
        "location": "warehouse_a",
        "extra_tag": "auto-labeled",
    }
)

# CR-1: assert which module the tensor-mode branch actually selected for the
# 7 blocks with a _tensor sibling, and that the 2 blocks with no sibling
# resolve to the same module regardless of mode.
PAIRED_IDENTIFIERS = {
    "roboflow_core/model_monitoring_inference_aggregator@v1",
    "roboflow_core/roboflow_custom_metadata@v1",
    "roboflow_core/roboflow_dataset_upload@v1",
    "roboflow_core/roboflow_dataset_upload@v2",
    "roboflow_core/roboflow_vision_events@v1",
    "roboflow_core/vision_event_bundle@v1",
    "roboflow_core/visual_search_classifier@v1",
}
UNPAIRED_IDENTIFIERS = {
    "roboflow_core/asset_library_attributes@v1",
    "roboflow_core/visual_search@v1",
}
modules_by_identifier = {
    b.manifest_type_identifier: b.fully_qualified_block_class_name
    for b in describe_available_blocks(dynamic_blocks=[]).blocks
    if b.manifest_type_identifier in PAIRED_IDENTIFIERS | UNPAIRED_IDENTIFIERS
}
assert (
    set(modules_by_identifier) == PAIRED_IDENTIFIERS | UNPAIRED_IDENTIFIERS
), modules_by_identifier
# `fully_qualified_block_class_name` is "<module path>.<ClassName>"; the
# module's own leaf (v1 / v1_tensor / v2 / v2_tensor) is the second-to-last
# dotted segment, not the last (which is the class name).
for identifier in PAIRED_IDENTIFIERS:
    module_name = modules_by_identifier[identifier]
    module_leaf = module_name.split(".")[-2]
    is_tensor_module = module_leaf.endswith("_tensor")
    assert is_tensor_module == requested_tensor_mode, (identifier, module_name)
for identifier in UNPAIRED_IDENTIFIERS:
    module_name = modules_by_identifier[identifier]
    module_leaf = module_name.split(".")[-2]
    assert module_leaf == "v1", (identifier, module_name)

print(
    json.dumps(
        {
            "outputs": result,
            "cache_is_injected": step._cache is cache,
            "cache_untouched": cache.storage == {},
        }
    )
)
""" % ("'''" + json.dumps(WORKFLOW) + "'''")

DISABLED_MESSAGE = "Sink was disabled by workflow execution policy"
EXPECTED = {
    "outputs": [{"message": DISABLED_MESSAGE}, {"message": DISABLED_MESSAGE}],
    "cache_is_injected": True,
    "cache_untouched": True,
}

RELOCATED_IDENTIFIERS = [
    "roboflow_core/asset_library_attributes@v1",
    "roboflow_core/model_monitoring_inference_aggregator@v1",
    "roboflow_core/roboflow_custom_metadata@v1",
    "roboflow_core/roboflow_dataset_upload@v1",
    "roboflow_core/roboflow_dataset_upload@v2",
    "roboflow_core/roboflow_vision_events@v1",
    "roboflow_core/vision_event_bundle@v1",
    "roboflow_core/visual_search@v1",
    "roboflow_core/visual_search_classifier@v1",
]

POLICY_PROBE = """
import json
import os

from inference.core.workflows.execution_engine.core import ExecutionEngine
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

RELOCATED = json.loads(%s)
available = {
    block.manifest_type_identifier
    for block in describe_available_blocks(dynamic_blocks=[]).blocks
}
WORKFLOW = {
    "version": "1.3.0",
    "inputs": [{"type": "WorkflowBatchInput", "name": "source_id", "kind": ["string"]}],
    "steps": [
        {
            "type": "roboflow_core/asset_library_attributes@v1",
            "name": "attrs",
            "source_id": "$inputs.source_id",
            "metadata": {},
            "tags": [],
            "disable_sink": False,
        }
    ],
    "outputs": [{"type": "JsonField", "name": "message", "selector": "$steps.attrs.message"}],
}
try:
    ExecutionEngine.init(
        workflow_definition=WORKFLOW, init_parameters={"workflows_core.api_key": "k"}
    )
    outcome = "compiled"
except Exception as error:
    outcome = type(error).__name__
print(
    json.dumps(
        {
            "present": sorted(i for i in RELOCATED if i in available),
            "asset_library_compile": outcome,
        }
    )
)
""" % ("'''" + json.dumps(RELOCATED_IDENTIFIERS) + "'''")

_VISUAL_SEARCH_ONLY = [
    "roboflow_core/visual_search@v1",
    "roboflow_core/visual_search_classifier@v1",
]
# (extra environment, identifiers that survive discovery, compile outcome) -
# every row measured at HEAD in both tensor modes (evidence E13).
POLICY_CASES = [
    (
        {"WORKFLOW_DISABLED_BLOCK_TYPES": "sink"},
        _VISUAL_SEARCH_ONLY,
        "WorkflowSyntaxError",
    ),
    # The OLD module path of the relocated sinks: an operator's existing
    # pattern must keep working after the move.
    (
        {"WORKFLOW_DISABLED_BLOCK_PATTERNS": "sinks.roboflow"},
        _VISUAL_SEARCH_ONLY,
        "WorkflowSyntaxError",
    ),
    # Class name (lower-cased) and display name.
    (
        {
            "WORKFLOW_DISABLED_BLOCK_PATTERNS": "roboflowdatasetuploadblockv1,visual_search"
        },
        [
            "roboflow_core/asset_library_attributes@v1",
            "roboflow_core/model_monitoring_inference_aggregator@v1",
            "roboflow_core/roboflow_custom_metadata@v1",
            "roboflow_core/roboflow_dataset_upload@v2",
            "roboflow_core/roboflow_vision_events@v1",
            "roboflow_core/vision_event_bundle@v1",
        ],
        "compiled",
    ),
    ({}, RELOCATED_IDENTIFIERS, "compiled"),
]


def _run(probe: str, tensor_mode: str, extra_env: dict) -> dict:
    env = {
        **os.environ,
        "DISABLE_VERSION_CHECK": "True",
        "ENABLE_TENSOR_DATA_REPRESENTATION": tensor_mode,
        # CR-1: the effective tensor mode is `ENABLE_TENSOR_DATA_REPRESENTATION
        # AND USE_INFERENCE_MODELS` (inference/core/env.py:1486); pin this so
        # "both modes" actually exercises the tensor code path rather than
        # running NumPy twice under an inherited USE_INFERENCE_MODELS=False.
        "USE_INFERENCE_MODELS": "True",
        "PYTHONPATH": os.pathsep.join(
            [str(REPO_ROOT), str(REPO_ROOT / "inference_models")]
        ),
    }
    # A case that sets neither variable must not inherit one from the shell.
    env.pop("WORKFLOW_DISABLED_BLOCK_TYPES", None)
    env.pop("WORKFLOW_DISABLED_BLOCK_PATTERNS", None)
    env.update(extra_env)
    result = subprocess.run(
        [sys.executable, "-c", probe], env=env, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("tensor_mode", ["False", "True"])
def test_roboflow_sink_workflow_is_unchanged_by_the_relocation(tensor_mode) -> None:
    assert _run(PROBE, tensor_mode, {}) == EXPECTED


@pytest.mark.parametrize("tensor_mode", ["False", "True"])
@pytest.mark.parametrize("extra_env,present,compile_outcome", POLICY_CASES)
def test_disable_policy_keeps_applying_to_the_relocated_blocks(
    tensor_mode, extra_env, present, compile_outcome
) -> None:
    assert _run(POLICY_PROBE, tensor_mode, extra_env) == {
        "present": present,
        "asset_library_compile": compile_outcome,
    }
