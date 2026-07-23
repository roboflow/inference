from typing import List, Literal, Type

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
)
from inference.core.workflows.execution_engine.v1.compiler.sink_disabling import (
    disable_workflow_sinks,
)
from inference.core.workflows.prototypes.block import (
    WorkflowBlock,
    WorkflowBlockManifest,
)


class NativeSinkManifest(WorkflowBlockManifest):
    type: Literal["test/native_sink@v1", "NativeSink"]
    disable_sink: bool = False

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class NativeSinkBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return NativeSinkManifest

    def run(self, disable_sink: bool) -> dict:
        return {}


class OrdinaryBlockManifest(WorkflowBlockManifest):
    type: Literal["test/ordinary@v1"]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return []


class OrdinaryBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return OrdinaryBlockManifest

    def run(self) -> dict:
        return {}


AVAILABLE_BLOCKS = [
    BlockSpecification(
        block_source="test",
        identifier="test.NativeSinkBlock",
        block_class=NativeSinkBlock,
        manifest_class=NativeSinkManifest,
    ),
    BlockSpecification(
        block_source="test",
        identifier="test.OrdinaryBlock",
        block_class=OrdinaryBlock,
        manifest_class=OrdinaryBlockManifest,
    ),
]


def test_native_sink_is_forced_into_its_noop_mode() -> None:
    workflow = _workflow(
        {"type": "test/native_sink@v1", "name": "sink", "disable_sink": False}
    )

    result = disable_workflow_sinks(workflow, available_blocks=AVAILABLE_BLOCKS)

    assert result.workflow_definition["steps"][0]["disable_sink"] is True
    assert result.steps_without_native_noop == frozenset()


def test_native_sink_selector_is_replaced_with_literal_true() -> None:
    workflow = _workflow(
        {
            "type": "NativeSink",
            "name": "sink",
            "disable_sink": "$inputs.disable_sink",
        }
    )

    result = disable_workflow_sinks(workflow, available_blocks=AVAILABLE_BLOCKS)

    assert result.workflow_definition["steps"][0]["disable_sink"] is True


def test_legacy_sink_without_noop_is_marked_for_execution_skip() -> None:
    workflow = _workflow(
        {"type": "roboflow_core/local_file_sink@v1", "name": "local_file"}
    )

    result = disable_workflow_sinks(workflow, available_blocks=AVAILABLE_BLOCKS)

    assert result.steps_without_native_noop == frozenset({"local_file"})
    assert result.workflow_definition == workflow


def test_legacy_sink_alias_is_marked_for_execution_skip() -> None:
    workflow = _workflow({"type": "RoboflowCustomMetadata", "name": "metadata"})

    result = disable_workflow_sinks(workflow, available_blocks=AVAILABLE_BLOCKS)

    assert result.steps_without_native_noop == frozenset({"metadata"})


def test_ordinary_block_is_not_changed_or_skipped() -> None:
    step = {"type": "test/ordinary@v1", "name": "transform"}
    workflow = _workflow(step)

    result = disable_workflow_sinks(workflow, available_blocks=AVAILABLE_BLOCKS)

    assert result.workflow_definition == workflow
    assert result.steps_without_native_noop == frozenset()


def test_disabling_sinks_does_not_mutate_caller_workflow() -> None:
    step = {"type": "test/native_sink@v1", "name": "sink", "disable_sink": False}
    workflow = _workflow(step)

    disable_workflow_sinks(workflow, available_blocks=AVAILABLE_BLOCKS)

    assert workflow["steps"][0]["disable_sink"] is False


def _workflow(step: dict) -> dict:
    return {
        "version": "1.0",
        "inputs": [],
        "steps": [step],
        "outputs": [],
    }
