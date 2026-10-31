"""Compile-time checks for `video_clip_anchor` (recorder -> sink wiring)."""

import warnings
from typing import List, Literal, Optional, Type, Union
from unittest import mock

import pytest
from pydantic import ConfigDict
from roboflow_workflows.errors import ExecutionGraphStructureError
from roboflow_workflows.execution_engine.entities.base import OutputDefinition
from roboflow_workflows.execution_engine.entities.types import (
    VIDEO_CLIP_ANCHOR_KIND,
    WILDCARD_KIND,
    Selector,
)
from roboflow_workflows.execution_engine.entities.workload import StepExecutionMode
from roboflow_workflows.execution_engine.v1.compiler import core as compiler_core
from roboflow_workflows.execution_engine.v1.compiler.entities import BlockSpecification
from roboflow_workflows.execution_engine.v1.compiler.video_clip_checks import (
    VideoClipRecorderUnderFlowControlWarning,
    find_anchor_carrying_selectors,
    verify_video_clip_anchors,
)
from roboflow_workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

RECORDER = {
    "type": "roboflow_core/video_clip_recorder@v1",
    "name": "recorder",
    "image": "$inputs.image",
}


class FakeClipSinkManifest(WorkflowBlockManifest):
    """Stands in for the Vision Events sink: declares the anchor kind on its input."""

    model_config = ConfigDict(json_schema_extra={"block_type": "sink"})
    type: Literal["test/fake_clip_sink@v1"]
    video_clips: Union[
        Selector(kind=[VIDEO_CLIP_ANCHOR_KIND]),
        List[Selector(kind=[VIDEO_CLIP_ANCHOR_KIND])],
    ]

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [OutputDefinition(name="videos", kind=[WILDCARD_KIND])]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.3.0,<2.0.0"


class FakeClipSinkBlock(WorkflowBlock):
    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return FakeClipSinkManifest

    def run(self, video_clips) -> BlockResult:
        return {"videos": []}


FAKE_SINK_SPECIFICATION = BlockSpecification(
    block_source="test_video_clip_checks",
    identifier="test.FakeClipSinkBlock",
    block_class=FakeClipSinkBlock,
    manifest_class=FakeClipSinkManifest,
)


@pytest.fixture(autouse=True)
def blocks_with_fake_sink():
    real_loader = compiler_core.load_workflow_blocks

    def load_with_fake_sink(*args, **kwargs):
        return real_loader(*args, **kwargs) + [FAKE_SINK_SPECIFICATION]

    with mock.patch.object(
        compiler_core, "load_workflow_blocks", side_effect=load_with_fake_sink
    ), mock.patch.object(compiler_core, "_is_resolver_dependent", return_value=True):
        yield


def _workflow(steps: list, outputs: list) -> dict:
    return {
        "version": "1.4.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": outputs,
    }


def _output(name: str, selector: str) -> dict:
    return {"type": "JsonField", "name": name, "selector": selector}


def _sink(anchors) -> dict:
    return {"type": "test/fake_clip_sink@v1", "name": "sink", "video_clips": anchors}


def _remote_inner_workflow(binding: str) -> dict:
    return {
        "type": "roboflow_core/inner_workflow@v1",
        "name": "remote_child",
        "execution_mode": "remote_dispatch",
        "workflow_definition": {
            "version": "1.4.0",
            "inputs": [{"type": "WorkflowParameter", "name": "anchor"}],
            "steps": [],
            "outputs": [],
        },
        "parameter_bindings": {"anchor": binding},
    }


def _frame_delay(data: str) -> dict:
    return {
        "type": "roboflow_core/frame_delay@v1",
        "name": "delay",
        "image": "$inputs.image",
        "data": data,
        "offset": -1,
    }


def _compile(definition: dict, init_parameters: Optional[dict] = None):
    return compiler_core.compile_workflow_graph(
        workflow_definition=definition,
        init_parameters=init_parameters or {},
    )


def test_recorder_wired_to_sink_compiles_without_warnings() -> None:
    # given
    definition = _workflow(
        steps=[RECORDER, _sink("$steps.recorder.anchor")],
        outputs=[_output("videos", "$steps.sink.videos")],
    )

    # when
    with warnings.catch_warnings():
        warnings.simplefilter("error", VideoClipRecorderUnderFlowControlWarning)
        result = _compile(definition)

    # then
    tainted = find_anchor_carrying_selectors(execution_graph=result.execution_graph)
    assert tainted == {"$steps.recorder.anchor"}, "the sink consumes, never forwards"


def test_list_of_anchors_from_two_recorders_compiles() -> None:
    # given
    annotated = {**RECORDER, "name": "annotated", "label": "Annotated"}
    definition = _workflow(
        steps=[
            RECORDER,
            annotated,
            _sink(["$steps.recorder.anchor", "$steps.annotated.anchor"]),
        ],
        outputs=[_output("videos", "$steps.sink.videos")],
    )

    # when
    result = _compile(definition)

    # then
    assert result.execution_graph is not None


@pytest.mark.parametrize("selector", ["$steps.recorder.anchor", "$steps.recorder.*"])
def test_anchor_as_workflow_output_is_rejected(selector: str) -> None:
    # given
    definition = _workflow(
        steps=[RECORDER, _sink("$steps.recorder.anchor")],
        outputs=[_output("leaked", selector)],
    )

    # when
    with pytest.raises(ExecutionGraphStructureError) as error:
        _compile(definition)

    # then
    message = error.value.public_message
    assert "Workflow output `leaked`" in message
    assert "video_clip_anchor" in message
    assert "cannot be returned from a workflow" in message


def test_anchor_into_remote_dispatch_inner_workflow_is_rejected() -> None:
    # given
    definition = _workflow(
        steps=[RECORDER, _remote_inner_workflow(binding="$steps.recorder.anchor")],
        outputs=[],
    )

    # when
    with pytest.raises(ExecutionGraphStructureError) as error:
        _compile(definition)

    # then
    message = error.value.public_message
    assert "Step `remote_child` binds `$steps.recorder.anchor`" in message
    assert "inner workflow with `remote_dispatch`" in message
    assert "cannot cross that boundary" in message


def test_anchor_passed_through_wildcard_block_is_still_rejected() -> None:
    # given - Frame Delay forwards whatever it receives as a wildcard output
    definition = _workflow(
        steps=[
            RECORDER,
            _frame_delay(data="$steps.recorder.anchor"),
            _remote_inner_workflow(binding="$steps.delay.output"),
        ],
        outputs=[],
    )

    # when
    with pytest.raises(ExecutionGraphStructureError) as error:
        _compile(definition)

    # then
    assert "binds `$steps.delay.output`" in error.value.public_message


def test_wildcard_taint_flags_pass_through_outputs_as_workflow_outputs() -> None:
    # given - known conservative behaviour, documented in video_clip_checks
    definition = _workflow(
        steps=[RECORDER, _frame_delay(data="$steps.recorder.anchor")],
        outputs=[_output("delayed", "$steps.delay.output")],
    )

    # when
    with pytest.raises(ExecutionGraphStructureError) as error:
        _compile(definition)

    # then
    assert "Workflow output `delayed`" in error.value.public_message


def test_untainted_frame_delay_output_is_accepted() -> None:
    # given - frame delay's non-wildcard outputs never carry the anchor
    definition = _workflow(
        steps=[
            RECORDER,
            _sink("$steps.recorder.anchor"),
            _frame_delay(data="$steps.recorder.anchor"),
        ],
        outputs=[_output("available", "$steps.delay.is_available")],
    )

    # when
    result = _compile(definition)

    # then
    tainted = find_anchor_carrying_selectors(execution_graph=result.execution_graph)
    assert tainted == {"$steps.recorder.anchor", "$steps.delay.output"}


def _buffer(data: str) -> dict:
    return {
        "type": "roboflow_core/buffer@v1",
        "name": "buffer",
        "data": data,
        "length": 5,
    }


def _dimension_collapse(data: str) -> dict:
    return {
        "type": "roboflow_core/dimension_collapse@v1",
        "name": "collapse",
        "data": data,
    }


@pytest.mark.parametrize(
    "container_step, selector",
    [
        (_buffer("$steps.recorder.anchor"), "$steps.buffer.output"),
        (_dimension_collapse("$steps.recorder.anchor"), "$steps.collapse.output"),
    ],
)
def test_anchor_through_list_output_as_workflow_output_is_rejected(
    container_step: dict, selector: str
) -> None:
    # given - Buffer and Dimension Collapse output `list_of_values`, not `*`
    definition = _workflow(
        steps=[RECORDER, container_step],
        outputs=[_output("anchors", selector)],
    )

    # when
    with pytest.raises(ExecutionGraphStructureError) as error:
        _compile(definition)

    # then
    message = error.value.public_message
    assert "Workflow output `anchors`" in message
    assert f"`{selector}`" in message


def test_anchor_through_buffer_into_remote_custom_python_is_rejected() -> None:
    # given
    definition = _workflow(
        steps=[
            RECORDER,
            _buffer("$steps.recorder.anchor"),
            {
                "type": "AnchorConsumer",
                "name": "consumer",
                "anchor": "$steps.buffer.output",
            },
        ],
        outputs=[],
    )
    definition["dynamic_blocks_definitions"] = [CUSTOM_PYTHON_DEFINITION]
    result = _compile(definition)

    # when
    with pytest.raises(ExecutionGraphStructureError) as error:
        verify_video_clip_anchors(
            result.execution_graph,
            available_blocks=result.available_blocks,
            custom_python_execution_mode="modal",
        )

    # then
    message = error.value.public_message
    assert "Step `consumer` binds `$steps.buffer.output` to `anchor`" in message


def test_buffer_of_non_anchor_data_next_to_a_recorder_compiles() -> None:
    # given - the normal recorder -> sink wiring beside an unrelated buffer
    definition = _workflow(
        steps=[RECORDER, _sink("$steps.recorder.anchor"), _buffer("$inputs.image")],
        outputs=[
            _output("videos", "$steps.sink.videos"),
            _output("frames", "$steps.buffer.output"),
        ],
    )

    # when
    result = _compile(definition)

    # then
    tainted = find_anchor_carrying_selectors(execution_graph=result.execution_graph)
    assert tainted == {"$steps.recorder.anchor"}


CUSTOM_PYTHON_DEFINITION = {
    "type": "DynamicBlockDefinition",
    "manifest": {
        "type": "ManifestDescription",
        "block_type": "AnchorConsumer",
        "inputs": {
            "anchor": {
                "type": "DynamicInputDefinition",
                "selector_types": ["step_output"],
            },
        },
        "outputs": {"result": {"type": "DynamicOutputDefinition", "kind": []}},
    },
    "code": {
        "type": "PythonCode",
        "run_function_code": "def run(self, anchor):\n    return {'result': None}\n",
    },
}


def _custom_python_workflow() -> dict:
    definition = _workflow(
        steps=[
            RECORDER,
            {
                "type": "AnchorConsumer",
                "name": "consumer",
                "anchor": "$steps.recorder.anchor",
            },
        ],
        outputs=[],
    )
    definition["dynamic_blocks_definitions"] = [CUSTOM_PYTHON_DEFINITION]
    return definition


def test_anchor_into_local_custom_python_is_accepted() -> None:
    # when
    result = _compile(_custom_python_workflow())

    # then
    verify_video_clip_anchors(
        result.execution_graph,
        available_blocks=result.available_blocks,
        custom_python_execution_mode="local",
    )


def test_anchor_into_remote_custom_python_is_rejected() -> None:
    # given - compiled locally (no Modal in unit tests), checked for `modal`
    result = _compile(_custom_python_workflow())

    # when
    with pytest.raises(ExecutionGraphStructureError) as error:
        verify_video_clip_anchors(
            result.execution_graph,
            available_blocks=result.available_blocks,
            custom_python_execution_mode="modal",
        )

    # then
    message = error.value.public_message
    assert "Step `consumer` binds `$steps.recorder.anchor` to `anchor`" in message
    assert "Custom Python with execution mode `modal`" in message


def test_anchor_into_remote_roboflow_model_step_is_rejected() -> None:
    # given - only a wildcard pass-through can reach a model's image input
    definition = _workflow(
        steps=[
            RECORDER,
            _frame_delay(data="$steps.recorder.anchor"),
            {
                "type": "roboflow_core/roboflow_object_detection_model@v2",
                "name": "model",
                "images": "$steps.delay.output",
                "model_id": "yolov8n-640",
            },
        ],
        outputs=[],
    )

    # when
    local = _compile(definition)
    with pytest.raises(ExecutionGraphStructureError) as error:
        _compile(
            definition,
            init_parameters={
                "workflows_core.step_execution_mode": StepExecutionMode.REMOTE
            },
        )

    # then
    assert local.execution_graph is not None
    assert "step execution mode `remote`" in error.value.public_message


def test_sink_step_is_not_treated_as_remote_model() -> None:
    # given - the Vision Events sink type also starts with `roboflow_core/roboflow_`
    definition = _workflow(
        steps=[RECORDER, _sink("$steps.recorder.anchor")],
        outputs=[_output("videos", "$steps.sink.videos")],
    )

    # when
    result = _compile(
        definition,
        init_parameters={"workflows_core.step_execution_mode": "remote"},
    )

    # then
    assert result.execution_graph is not None


def test_recorder_under_continue_if_warns() -> None:
    # given
    gate = {
        "type": "roboflow_core/continue_if@v1",
        "name": "gate",
        "condition_statement": {
            "type": "StatementGroup",
            "statements": [
                {
                    "type": "BinaryStatement",
                    "left_operand": {"type": "StaticOperand", "value": 1},
                    "comparator": {"type": "(Number) =="},
                    "right_operand": {"type": "StaticOperand", "value": 1},
                }
            ],
        },
        "next_steps": ["$steps.recorder"],
        "evaluation_parameters": {"image": "$inputs.image"},
    }
    definition = _workflow(
        steps=[gate, RECORDER, _sink("$steps.recorder.anchor")],
        outputs=[_output("videos", "$steps.sink.videos")],
    )

    # when
    with pytest.warns(VideoClipRecorderUnderFlowControlWarning) as record:
        _compile(definition)

    # then
    messages = [str(warning.message) for warning in record]
    assert any(
        "Step `recorder` (Video Clip Recorder) runs under flow control" in message
        and "gate the sink instead" in message
        for message in messages
    )
