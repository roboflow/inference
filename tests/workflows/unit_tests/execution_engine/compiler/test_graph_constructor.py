from typing import List

import pytest

from inference.enterprise.workflows.entities.base import (
    InferenceImage,
    InferenceParameter,
    JsonField,
)
from inference.enterprise.workflows.entities.types import (
    INTEGER_KIND,
    ROBOFLOW_MODEL_ID_KIND,
)
from inference.enterprise.workflows.errors import (
    ConditionalBranchesCollapseError,
    DanglingExecutionBranchError,
    ExecutionGraphStructureError,
    InvalidReferenceTargetError,
    ReferenceTypeError,
)
from inference.enterprise.workflows.execution_engine.compiler.entities import (
    BlockSpecification,
    ParsedWorkflowDefinition,
)
from inference.enterprise.workflows.execution_engine.compiler.graph_constructor import (
    prepare_execution_graph,
)
from inference.enterprise.workflows.execution_engine.compiler.utils import (
    FLOW_CONTROL_NODE_KEY,
)
from tests.workflows.unit_tests.execution_engine.compiler.plugin_with_test_blocks.blocks import (
    ExampleFlowControlBlock,
    ExampleFlowControlBlockManifest,
    ExampleFusionBlock,
    ExampleFusionBlockManifest,
    ExampleModelBlock,
    ExampleModelBlockManifest,
    ExampleSinkBlock,
    ExampleTransformationBlock,
    ExampleTransformationBlockManifest,
)


@pytest.fixture(scope="function")
def blocks_from_test_plugin() -> List[BlockSpecification]:
    types = [
        ExampleModelBlock,
        ExampleFlowControlBlock,
        ExampleTransformationBlock,
        ExampleSinkBlock,
        ExampleFusionBlock,
    ]
    return [
        BlockSpecification(
            block_source="test_plugin",
            identifier=f"test_plugin.{t.__name__}",
            block_class=t,
            manifest_class=t.get_input_manifest(),
        )
        for t in types
    ]


def test_execution_graph_construction_for_trivial_workflow(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[InferenceImage(type="InferenceImage", name="image")],
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
    result = prepare_execution_graph(
        workflow_definition=workflow_definition,
        available_blocks=blocks_from_test_plugin,
    )

    # then
    assert (
        len(result.nodes) == 3
    ), "Expected 1 input node, 1 step node and one output node"
    assert (
        result.nodes["$inputs.image"]["definition"].name == "image"
    ), "Image node must be named correctly"
    assert (
        result.nodes["$steps.model_1"]["definition"].name == "model_1"
    ), "Model node must be named correctly"
    assert (
        result.nodes["$outputs.predictions"]["definition"].name == "predictions"
    ), "Output node must be named correctly"
    assert result.has_edge(
        "$inputs.image", "$steps.model_1"
    ), "Input image must be connected to model step"
    assert result.has_edge(
        "$steps.model_1", "$outputs.predictions"
    ), "Model step must be connected to output"


def test_execution_graph_construction_when_there_is_input_selector_missing(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
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
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_output_selector_points_non_existing_step(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[InferenceImage(type="InferenceImage", name="image")],
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
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_output_defines_non_existing_output(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image"),
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
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_there_is_a_dangling_output(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[InferenceImage(type="InferenceImage", name="image")],
        steps=[
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1",
                image="$inputs.image",
                model_id="my_model",
            )
        ],
        outputs=[],
    )

    # when
    with pytest.raises(DanglingExecutionBranchError):
        # TODO: consider if that's actually good to raise error in this case
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_input_kind_does_not_match_block_manifest(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image"),
            InferenceParameter(
                type="InferenceParameter", name="model", kind=[INTEGER_KIND]
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
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_selector_is_injected_to_string_field(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image"),
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
        available_blocks=blocks_from_test_plugin,
    )

    # then - we just check that selector not defined in manifest was ignored
    assert (
        len(result.nodes) == 3
    ), "Expected 1 input node, 1 step node and one output node"


def test_execution_graph_construction_when_two_parallel_execution_branches_exists(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image_1"),
            InferenceImage(type="InferenceImage", name="image_2"),
            InferenceParameter(
                type="InferenceParameter", name="model", kind=[ROBOFLOW_MODEL_ID_KIND]
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
        available_blocks=blocks_from_test_plugin,
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


def test_execution_graph_construction_when_fusion_of_two_branches_is_present(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image_1"),
            InferenceImage(type="InferenceImage", name="image_2"),
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
        available_blocks=blocks_from_test_plugin,
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


def test_execution_graph_construction_when_there_is_flow_control_step(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image"),
        ],
        steps=[
            ExampleFlowControlBlockManifest(
                type="ExampleFlowControl",
                name="random_choice",
                steps_to_choose=["$steps.model_1", "$steps.model_2"],
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
        available_blocks=blocks_from_test_plugin,
    )

    # then
    assert (
        len(result.nodes) == 6
    ), "Expected 1 input node, 3 step nodes and 2 output nodes"
    assert (
        "$inputs.image" in result.nodes
    ), "Expected image input to be a node in execution graph"
    assert (
        "$steps.random_choice" in result.nodes
    ), "Expected random_choice step to be a node in execution graph"
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
    assert len(result.edges) == 6, "Only 6 unique edges expected in the graph"
    assert result.has_edge(
        "$inputs.image", "$steps.model_1"
    ), "Expected to see connection between image and model_1"
    assert result.has_edge(
        "$steps.random_choice", "$steps.model_1"
    ), "Expected to see connection between random_choice and model_1"
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
    assert (
        result.nodes["$steps.random_choice"][FLOW_CONTROL_NODE_KEY] is True
    ), "Expected random_choice step to be recognised as control flow"


def test_execution_graph_construction_when_there_is_condition_branches_collapse(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image"),
        ],
        steps=[
            ExampleFlowControlBlockManifest(
                type="ExampleFlowControl",
                name="random_choice",
                steps_to_choose=["$steps.model_1", "$steps.model_2"],
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
            ExampleFusionBlockManifest(  # this step causes collapse
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
    with pytest.raises(ConditionalBranchesCollapseError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_there_is_collapse_of_two_conditional_branches_originated_in_different_root(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image_1"),
            InferenceImage(type="InferenceImage", name="image_2"),
        ],
        steps=[
            ExampleFlowControlBlockManifest(
                type="ExampleFlowControl",
                name="random_choice_1",
                steps_to_choose=["$steps.model_1a", "$steps.model_1b"],
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1a",
                image="$inputs.image_1",
                model_id="my_model",
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_1b",
                image="$inputs.image_1",
                model_id="my_model",
            ),
            ExampleFlowControlBlockManifest(
                type="ExampleFlowControl",
                name="random_choice_2",
                steps_to_choose=["$steps.model_2a", "$steps.model_2b"],
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_2a",
                image="$inputs.image_2",
                model_id="my_model",
            ),
            ExampleModelBlockManifest(
                type="ExampleModel",
                name="model_2b",
                image="$inputs.image_2",
                model_id="my_model",
            ),
            ExampleFusionBlockManifest(  # this step causes collapse
                type="ExampleFusion",
                name="fusion_a",
                predictions=[
                    "$steps.model_1a.predictions",
                    "$steps.model_2a.predictions",
                ],
            ),
            ExampleFusionBlockManifest(  # this step causes collapse
                type="ExampleFusion",
                name="fusion_b",
                predictions=[
                    "$steps.model_1b.predictions",
                    "$steps.model_2b.predictions",
                ],
            ),
        ],
        outputs=[
            JsonField(
                type="JsonField",
                name="predictions_a",
                selector="$steps.fusion_a.predictions",
            ),
            JsonField(
                type="JsonField",
                name="predictions_b",
                selector="$steps.fusion_b.predictions",
            ),
        ],
    )

    # when
    with pytest.raises(ConditionalBranchesCollapseError):
        _ = prepare_execution_graph(
            workflow_definition=workflow_definition,
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_cycle_is_detected(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image"),
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
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_reference_to_non_existing_step_output_provided(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image"),
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
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_reference_to_non_existing_step_provided(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image"),
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
            available_blocks=blocks_from_test_plugin,
        )


def test_execution_graph_construction_when_connection_kind_missmatch_detected(
    blocks_from_test_plugin: List[BlockSpecification],
) -> None:
    # given
    workflow_definition = ParsedWorkflowDefinition(
        version="1.0",
        inputs=[
            InferenceImage(type="InferenceImage", name="image"),
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
            available_blocks=blocks_from_test_plugin,
        )
