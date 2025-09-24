import pytest

from inference.core.workflows.errors import (
    ExecutionGraphStructureError,
    InvalidReferenceTargetError,
    ReferenceTypeError,
)
from inference.core.workflows.execution_engine.entities.base import (
    JsonField,
    WorkflowImage,
    WorkflowParameter,
)
from inference.core.workflows.execution_engine.entities.types import (
    INTEGER_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_MODEL_ID_KIND,
)
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    DynamicStepInputDefinition,
    InputNode,
    NodeCategory,
    NodeInputCategory,
    OutputNode,
    ParameterSpecification,
    ParsedWorkflowDefinition,
    StaticStepInputDefinition,
    StepNode,
)
from inference.core.workflows.execution_engine.v1.compiler.graph_constructor import (
    prepare_execution_graph,
)
from tests.workflows.unit_tests.execution_engine.compiler.plugin_with_test_blocks.blocks import (
    ExampleFlowControlBlockManifest,
    ExampleFusionBlockManifest,
    ExampleModelBlockManifest,
    ExampleNonBatchFlowControlBlockManifest,
    ExampleTransformationBlockManifest,
)


def test_execution_graph_construction_for_trivial_workflow() -> None:
    # given
    input_manifest = WorkflowImage(type="WorkflowImage", name="image")
    step_manifest = ExampleModelBlockManifest(
        type="ExampleModel",
        name="model_1",
        images="$inputs.image",
        model_id="my_model",
    )
    output_manifest = JsonField(
        type="JsonField",
        name="predictions",
        selector="$steps.model_1.predictions",
    )
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[input_manifest],
        steps=[step_manifest],
        outputs=[output_manifest],
    )

    # when
    result = prepare_execution_graph(
        workflow_definition=workflow_definition,
    )

    # then
    assert (
        len(result.nodes) == 3
    ), "Expected 1 input node, 1 step node and one output node"
    input_node = result.nodes["$inputs.image"]["node_compilation_output"]
    assert input_node == InputNode(
        node_category=NodeCategory.INPUT_NODE,
        name="image",
        selector="$inputs.image",
        data_lineage=["<workflow_input>"],
        input_manifest=input_manifest,
    ), "Image node must be created correctly"
    step_node = result.nodes["$steps.model_1"]["node_compilation_output"]
    assert step_node == StepNode(
        node_category=NodeCategory.STEP_NODE,
        name="model_1",
        selector="$steps.model_1",
        data_lineage=["<workflow_input>"],
        step_manifest=step_manifest,
        input_data={
            "images": DynamicStepInputDefinition(
                parameter_specification=ParameterSpecification(
                    parameter_name="images",
                    nested_element_key=None,
                    nested_element_index=None,
                ),
                category=NodeInputCategory.BATCH_INPUT_PARAMETER,
                data_lineage=["<workflow_input>"],
                selector="$inputs.image",
            ),
            "model_id": StaticStepInputDefinition(
                parameter_specification=ParameterSpecification(
                    parameter_name="model_id",
                    nested_element_key=None,
                    nested_element_index=None,
                ),
                category=NodeInputCategory.STATIC_VALUE,
                value="my_model",
            ),
            "string_value": StaticStepInputDefinition(  # default value provided at step level
                parameter_specification=ParameterSpecification(
                    parameter_name="string_value",
                    nested_element_key=None,
                    nested_element_index=None,
                ),
                category=NodeInputCategory.STATIC_VALUE,
                value=None,
            ),
        },
        batch_oriented_parameters={"images"},
        step_execution_dimensionality=1,
    ), "Model node must be created correctly"
    output_node = result.nodes["$outputs.predictions"]["node_compilation_output"]
    assert output_node == OutputNode(
        node_category=NodeCategory.OUTPUT_NODE,
        name="predictions",
        selector="$outputs.predictions",
        data_lineage=["<workflow_input>"],
        output_manifest=output_manifest,
        kind=[OBJECT_DETECTION_PREDICTION_KIND],
    ), "Output node must be created correctly"
    assert result.has_edge(
        "$inputs.image", "$steps.model_1"
    ), "Input image must be connected to model step"
    assert result.has_edge(
        "$steps.model_1", "$outputs.predictions"
    ), "Model step must be connected to output"


def test_execution_graph_construction_when_there_is_input_selector_missing() -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="my_model",
            )
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.model_1.predictions",
            )
        ],
    )

    # when
    with pytest.raises(InvalidReferenceTargetError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
        )


def test_execution_graph_construction_when_output_selector_points_non_existing_step() -> (
    None
):
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[WorkflowImage(type="WorkflowImage", name="image")],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="my_model",
            )
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.model_2.predictions",
            )
        ],
    )

    # when
    with pytest.raises(InvalidReferenceTargetError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
        )


def test_execution_graph_construction_when_output_defines_non_existing_output() -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image"),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="my_model",
            )
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.model_1.non_existing",
            )
        ],
    )

    # when
    with pytest.raises(InvalidReferenceTargetError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
        )


def test_execution_graph_construction_when_input_kind_does_not_match_block_manifest() -> (
    None
):
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image"),
            WorkflowParameter(
                type="WorkflowParameter", name="model", kind=[INTEGER_KIND]
            ),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="$inputs.model",
            )
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.model_1.predictions",
            )
        ],
    )

    # when
    with pytest.raises(ReferenceTypeError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
        )


def test_execution_graph_construction_when_selector_is_injected_to_string_field() -> (
    None
):
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image"),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="my_model",
                string_value="$inputs.faulty",
            )
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.model_1.predictions",
            )
        ],
    )

    # when
    result = prepare_execution_graph(
        workflow_definition=workflow_definition,
    )

    # then - we just check that selector not defined in manifest was ignored
    assert (
        len(result.nodes) == 3
    ), "Expected 1 input node, 1 step node and one output node"


def test_execution_graph_construction_when_two_parallel_execution_branches_exists() -> (
    None
):
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image_1"),
            WorkflowImage(type="WorkflowImage", name="image_2"),
            WorkflowParameter(
                type="WorkflowParameter", name="model", kind=[ROBOFLOW_MODEL_ID_KIND]
            ),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image_1",
                model_id="$inputs.model",
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_2",
                image="$inputs.image_2",
                model_id="my_model",
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
    result = prepare_execution_graph(
        workflow_definition=workflow_definition,
    )

    # then
    assert (
        len(result.nodes) == 7
    ), "Expected 3 input nodes, 2 step nodes and 2 output nodes"
    assert (
        "$inputs.image_1" in result.nodes
    ), "Expected image_1 input to be a node in execution graph"
    assert (
        "$inputs.image_2" in result.nodes
    ), "Expected image_2 input to be a node in execution graph"
    assert (
        "$inputs.model" in result.nodes
    ), "Expected model input to be a node in execution graph"
    assert (
        "$steps.model_1" in result.nodes
    ), "Expected model_1 step to be a node in execution graph"
    assert (
        "$steps.model_2" in result.nodes
    ), "Expected model_2 step to be a node in execution graph"
    assert (
        "$outputs.predictions_1" in result.nodes
    ), "Expected predictions_1 output to be a node in execution graph"
    assert (
        "$outputs.predictions_2" in result.nodes
    ), "Expected predictions_2 output to be a node in execution graph"
    assert len(result.edges) == 5, "Only 5 unique edges expected in the graph"
    assert result.has_edge(
        "$inputs.image_1", "$steps.model_1"
    ), "Expected to see connection between image_1 and model_1"
    assert result.has_edge(
        "$inputs.model", "$steps.model_1"
    ), "Expected to see connection between model input param and model_1"
    assert result.has_edge(
        "$inputs.image_2", "$steps.model_2"
    ), "Expected to see connection between image_2 and model_2"
    assert result.has_edge(
        "$steps.model_1", "$outputs.predictions_1"
    ), "Expected to see connection between model_1 and predictions_1 output"
    assert result.has_edge(
        "$steps.model_2", "$outputs.predictions_2"
    ), "Expected to see connection between model_2 and predictions_2 output"


def test_execution_graph_construction_when_fusion_of_two_branches_is_present() -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image_1"),
            WorkflowImage(type="WorkflowImage", name="image_2"),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image_1",
                model_id="my_model",
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_2",
                image="$inputs.image_2",
                model_id="my_model",
            ),
            ExampleFusionBlockManifest(
                type="ExampleFusion",
                name="fusion",
                predictions=[
                    "$steps.model_1.predictions",
                    "$steps.model_2.predictions",
                ],
            ),
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.fusion.predictions",
            ),
        ],
    )

    # when
    result = prepare_execution_graph(
        workflow_definition=workflow_definition,
    )

    # then
    assert (
        len(result.nodes) == 6
    ), "Expected 2 input nodes, 3 step nodes and 1 output node"
    assert (
        "$inputs.image_1" in result.nodes
    ), "Expected image_1 input to be a node in execution graph"
    assert (
        "$inputs.image_2" in result.nodes
    ), "Expected image_2 input to be a node in execution graph"
    assert (
        "$steps.model_1" in result.nodes
    ), "Expected model_1 step to be a node in execution graph"
    assert (
        "$steps.model_2" in result.nodes
    ), "Expected model_2 step to be a node in execution graph"
    assert (
        "$steps.fusion" in result.nodes
    ), "Expected fusion step to be a node in execution graph"
    assert (
        "$outputs.predictions" in result.nodes
    ), "Expected predictions output to be a node in execution graph"
    assert len(result.edges) == 5, "Only 5 unique edges expected in the graph"
    assert result.has_edge(
        "$inputs.image_1", "$steps.model_1"
    ), "Expected to see connection between image_1 and model_1"
    assert result.has_edge(
        "$inputs.image_2", "$steps.model_2"
    ), "Expected to see connection between image_2 and model_2"
    assert result.has_edge(
        "$steps.model_1", "$steps.fusion"
    ), "Expected to see connection between step_1 and fusion"
    assert result.has_edge(
        "$steps.model_2", "$steps.fusion"
    ), "Expected to see connection between step_2 and fusion"
    assert result.has_edge(
        "$steps.fusion", "$outputs.predictions"
    ), "Expected to see connection between fusion step and predictions output"


def test_execution_graph_construction_when_there_is_flow_control_step() -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image"),
        ],
        steps=[
            ExampleFlowControlBlockManifest(
                type="ExampleFlowControl",
                name="random_choice",
                a_steps=["$steps.model_1"],
                b_steps=["$steps.model_2"],
            ),
            ExampleNonBatchFlowControlBlockManifest(
                type="ExampleNonBatchFlowControl",
                name="non_batch_condition",
                next_steps=["$steps.model_1"],
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="my_model",
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_2",
                image="$inputs.image",
                model_id="my_model",
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
    result = prepare_execution_graph(
        workflow_definition=workflow_definition,
    )

    # then
    assert (
        len(result.nodes) == 7
    ), "Expected 1 input node, 3 step nodes and 2 output nodes"
    assert (
        "$inputs.image" in result.nodes
    ), "Expected image input to be a node in execution graph"
    assert (
        "$steps.random_choice" in result.nodes
    ), "Expected random_choice step to be a node in execution graph"
    assert result.nodes["$steps.random_choice"][
        "node_compilation_output"
    ].child_execution_branches == {
        "$steps.model_1": f"Branch[$steps.random_choice -> a_steps]",
        "$steps.model_2": f"Branch[$steps.random_choice -> b_steps]",
    }, "Expected execution branches to be denoted properly for random_choice step"
    assert (
        "$steps.non_batch_condition" in result.nodes
    ), "Expected non_batch_condition step node in execution graph"
    assert result.nodes["$steps.non_batch_condition"][
        "node_compilation_output"
    ].child_execution_branches == {
        "$steps.model_1": f"Branch[$steps.non_batch_condition -> next_steps]",
    }, "Expected execution branches to be denoted properly for non_batch_condition step"
    assert result.nodes["$steps.model_1"][
        "node_compilation_output"
    ].execution_branches_impacting_inputs == {
        f"Branch[$steps.random_choice -> a_steps]",
        f"Branch[$steps.non_batch_condition -> next_steps]",
    }, "Expected execution branches impacting inputs to be denoted"
    assert result.nodes["$steps.model_2"][
        "node_compilation_output"
    ].execution_branches_impacting_inputs == {
        f"Branch[$steps.random_choice -> b_steps]",
    }
    assert (
        "$steps.model_1" in result.nodes
    ), "Expected model_1 step to be a node in execution graph"
    assert (
        "$steps.model_2" in result.nodes
    ), "Expected model_2 step to be a node in execution graph"
    assert (
        "$outputs.predictions_1" in result.nodes
    ), "Expected predictions_1 output to be a node in execution graph"
    assert (
        "$outputs.predictions_2" in result.nodes
    ), "Expected predictions_2 output to be a node in execution graph"
    assert len(result.edges) == 7, "Only 7 unique edges expected in the graph"
    assert result.has_edge(
        "$inputs.image", "$steps.model_1"
    ), "Expected to see connection between image and model_1"
    assert result.has_edge(
        "$steps.random_choice", "$steps.model_1"
    ), "Expected to see connection between random_choice and model_1"
    assert result.has_edge(
        "$steps.non_batch_condition", "$steps.model_1"
    ), "Expected to see connection between non_batch_condition and model_1"
    assert result.has_edge(
        "$inputs.image", "$steps.model_2"
    ), "Expected to see connection between image and model_2"
    assert result.has_edge(
        "$steps.random_choice", "$steps.model_2"
    ), "Expected to see connection between random_choice and model_2"
    assert result.has_edge(
        "$steps.model_1", "$outputs.predictions_1"
    ), "Expected to see connection between model_1 and predictions_1"
    assert result.has_edge(
        "$steps.model_2", "$outputs.predictions_2"
    ), "Expected to see connection between model_2 and predictions_2"
    assert result.nodes["$steps.random_choice"][
        "node_compilation_output"
    ].controls_flow(), "Expected random_choice step to be recognised as control flow"


def test_execution_graph_construction_when_cycle_is_detected() -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image"),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$steps.transformation.image",
                model_id="my_model",
            ),
            ExampleTransformationBlockManifest(
                type="ExampleTransformation",
                name="transformation",
                image="$inputs.image",
                predictions="$steps.model_1.predictions",
            ),
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.transformation.predictions",
            ),
        ],
    )

    # when
    with pytest.raises(ExecutionGraphStructureError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
        )


def test_execution_graph_construction_when_reference_to_non_existing_step_output_provided() -> (
    None
):
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image"),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="my_model",
            ),
            ExampleTransformationBlockManifest(
                type="ExampleTransformation",
                name="transformation",
                image="$inputs.image",
                predictions="$steps.model_1.invalid",
            ),
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.transformation.predictions",
            ),
        ],
    )

    # when
    with pytest.raises(ExecutionGraphStructureError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
        )


def test_execution_graph_construction_when_reference_to_non_existing_step_provided() -> (
    None
):
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image"),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="my_model",
            ),
            ExampleTransformationBlockManifest(
                type="ExampleTransformation",
                name="transformation",
                image="$inputs.image",
                predictions="$steps.invalid.predictions",
            ),
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.transformation.predictions",
            ),
        ],
    )

    # when
    with pytest.raises(InvalidReferenceTargetError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
        )


def test_execution_graph_construction_when_connection_kind_missmatch_detected() -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            WorkflowImage(type="WorkflowImage", name="image"),
        ],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="my_model",
            ),
            ExampleTransformationBlockManifest(
                type="ExampleTransformation",
                name="transformation",
                image="$steps.model_1.predictions",
                predictions="$steps.model_1.predictions",
            ),
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions",
                selector="$steps.transformation.predictions",
            ),
        ],
    )

    # when
    with pytest.raises(ReferenceTypeError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
        )
