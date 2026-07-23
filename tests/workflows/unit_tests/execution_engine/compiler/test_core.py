from collections import defaultdict
from unittest import mock

from inference.core.workflows.execution_engine.entities.base import (
    JsonField,
    WorkflowImage,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.v1.compiler.core import (
    collect_input_substitutions,
    compile_workflow_graph,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    BlockSpecification,
    ParsedWorkflowDefinition,
)
from tests.workflows.unit_tests.execution_engine.compiler.plugin_with_test_blocks.blocks import (
    ExampleModelBlock,
    ExampleModelBlockManifest,
)


def test_collect_input_substitutions() -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image_1"),
            WorkflowImage(type="WorkflowImage", name="image_2"),
            WorkflowParameter(type="WorkflowParameter", name="model_1"),
            WorkflowParameter(type="WorkflowParameter", name="model_2"),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image_1",
                model_id="$inputs.model_1",
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_2",
                image="$inputs.image_2",
                model_id="$inputs.model_2",
            ),
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions_1",
                selector="$steps.model_1.predictions",
            ),
            JsonField(
                type="JsonField",
                name="predictions_2",
                selector="$steps.model_2.predictions",
            ),
        ],
    )

    # when
    result = collect_input_substitutions(workflow_definition=workflow_definition)
    aggregated_results = defaultdict(dict)
    for element in result:
        aggregated_results[id(element.step_manifest)][
            element.manifest_property
        ] = element.input_parameter_name

    # then
    assert aggregated_results == {
        id(workflow_definition.steps[0]): {
            "model_id": "model_1",
        },
        id(workflow_definition.steps[1]): {
            "model_id": "model_2",
        },
    }


@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.validate_workflow_specification"
)
@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.prepare_execution_graph"
)
@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.parse_workflow_definition"
)
@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.inline_inner_workflow_steps"
)
@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.validate_inner_workflow_composition_from_raw_workflow_definition"
)
@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.compile_dynamic_blocks",
    return_value=[],
)
@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.load_kinds_deserializers",
    return_value={},
)
@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.load_kinds_serializers",
    return_value={},
)
@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.load_initializers",
    return_value={},
)
@mock.patch(
    "inference.core.workflows.execution_engine.v1.compiler.core.load_workflow_blocks"
)
def test_sink_disabling_is_applied_after_inner_workflow_inlining(
    load_workflow_blocks,
    _load_initializers,
    _load_kinds_serializers,
    _load_kinds_deserializers,
    _compile_dynamic_blocks,
    _validate_inner_workflow,
    inline_inner_workflow_steps,
    parse_workflow_definition,
    prepare_execution_graph,
    _validate_workflow,
) -> None:
    class NativeSinkManifest(ExampleModelBlockManifest):
        disable_sink: bool = False

    native_sink = BlockSpecification(
        block_source="test",
        identifier="test.NativeSink",
        block_class=ExampleModelBlock,
        manifest_class=NativeSinkManifest,
    )
    load_workflow_blocks.return_value = [native_sink]
    inline_inner_workflow_steps.return_value = {
        "version": "1.0",
        "inputs": [],
        "steps": [
            {
                "type": "ExampleModel",
                "name": "inner__sink",
                "disable_sink": False,
            }
        ],
        "outputs": [],
    }
    parsed = mock.MagicMock(steps=[], inputs=[], outputs=[])
    parse_workflow_definition.return_value = parsed
    prepare_execution_graph.return_value = mock.MagicMock()

    result = compile_workflow_graph(
        workflow_definition={
            "version": "1.0",
            "id": "sink-disabling-after-inlining-test",
            "inputs": [],
            "steps": [],
            "outputs": [],
        },
        disable_sinks=True,
    )

    parsed_definition = parse_workflow_definition.call_args.kwargs[
        "raw_workflow_definition"
    ]
    assert parsed_definition["steps"][0]["disable_sink"] is True
    assert result.disabled_steps == frozenset()
