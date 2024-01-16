import networkx as nx

from inference.enterprise.deployments.complier.graph_parser import (
    add_input_nodes_for_graph,
    add_output_nodes_for_graph,
    add_steps_nodes_for_graph,
    construct_graph,
)
from inference.enterprise.deployments.constants import (
    INPUT_NODE_KIND,
    OUTPUT_NODE_KIND,
    STEP_NODE_KIND,
)
from inference.enterprise.deployments.entities.deployment_specs import DeploymentSpecV1
from inference.enterprise.deployments.entities.inputs import (
    InferenceImage,
    InferenceParameter,
)
from inference.enterprise.deployments.entities.outputs import JsonField
from inference.enterprise.deployments.entities.steps import Crop, ObjectDetectionModel


def test_add_input_nodes_for_graph() -> None:
    # given
    execution_graph = nx.DiGraph()
    inputs = [
        InferenceImage(type="InferenceImage", name="image"),
        InferenceParameter(type="InferenceParameter", name="x"),
        InferenceParameter(type="InferenceParameter", name="y"),
    ]

    # when
    execution_graph = add_input_nodes_for_graph(
        inputs=inputs,
        execution_graph=execution_graph,
    )

    # then
    assert execution_graph.nodes["$inputs.image"]["kind"] == INPUT_NODE_KIND
    assert execution_graph.nodes["$inputs.image"]["definition"] == inputs[0]
    assert execution_graph.nodes["$inputs.x"]["kind"] == INPUT_NODE_KIND
    assert execution_graph.nodes["$inputs.x"]["definition"] == inputs[1]
    assert execution_graph.nodes["$inputs.y"]["kind"] == INPUT_NODE_KIND
    assert execution_graph.nodes["$inputs.y"]["definition"] == inputs[2]


def test_add_steps_nodes_for_graph() -> None:
    # given
    execution_graph = nx.DiGraph()
    steps = [
        Crop(
            type="Crop",
            name="my_crop",
            image="$inputs.image",
            detections="$steps.detect_2.predictions",
        ),
        ObjectDetectionModel(
            type="ObjectDetectionModel",
            name="my_model",
            image="$inputs.image",
            model_id="some/1",
            confidence=0.3,
        ),
    ]

    # when
    execution_graph = add_steps_nodes_for_graph(
        steps=steps,
        execution_graph=execution_graph,
    )

    # then
    assert execution_graph.nodes["$steps.my_crop"]["kind"] == STEP_NODE_KIND
    assert execution_graph.nodes["$steps.my_crop"]["definition"] == steps[0]
    assert execution_graph.nodes["$steps.my_model"]["kind"] == STEP_NODE_KIND
    assert execution_graph.nodes["$steps.my_model"]["definition"] == steps[1]


def test_add_output_nodes_for_graph() -> None:
    # given
    execution_graph = nx.DiGraph()
    outputs = [
        JsonField(type="JsonField", name="some", selector="$steps.a.predictions"),
        JsonField(type="JsonField", name="other", selector="$steps.b.predictions"),
    ]

    # when
    execution_graph = add_output_nodes_for_graph(
        outputs=outputs,
        execution_graph=execution_graph,
    )

    # then
    assert execution_graph.nodes["$outputs.some"]["kind"] == OUTPUT_NODE_KIND
    assert execution_graph.nodes["$outputs.some"]["definition"] == outputs[0]
    assert execution_graph.nodes["$outputs.other"]["kind"] == OUTPUT_NODE_KIND
    assert execution_graph.nodes["$outputs.other"]["definition"] == outputs[1]


def test_construct_graph() -> None:
    # given
    deployment_specs = DeploymentSpecV1.parse_obj(
        {
            "version": "1.0",
            "inputs": [
                {"type": "InferenceImage", "name": "image"},
                {"type": "InferenceParameter", "name": "confidence"},
            ],
            "steps": [
                {
                    "type": "ClassificationModel",
                    "name": "step_1",
                    "image": "$inputs.image",
                    "model_id": "vehicle-classification-eapcd/2",
                    "confidence": "$inputs.confidence",
                },
                {
                    "type": "Condition",
                    "name": "step_2",
                    "left": "$steps.step_1.top",
                    "operator": "equal",
                    "right": "Car",
                    "step_if_true": "$steps.step_3",
                    "step_if_false": "$steps.step_4",
                },
                {
                    "type": "ObjectDetectionModel",
                    "name": "step_3",
                    "image": "$inputs.image",
                    "model_id": "yolov8n-640",
                    "confidence": 0.5,
                    "iou_threshold": 0.4,
                },
                {
                    "type": "ObjectDetectionModel",
                    "name": "step_4",
                    "image": "$inputs.image",
                    "model_id": "yolov8n-1280",
                    "confidence": 0.5,
                    "iou_threshold": 0.4,
                },
            ],
            "outputs": [
                {
                    "type": "JsonField",
                    "name": "top_class",
                    "selector": "$steps.step_1.top",
                },
                {
                    "type": "JsonField",
                    "name": "step_3_predictions",
                    "selector": "$steps.step_3.predictions",
                },
                {
                    "type": "JsonField",
                    "name": "step_4_predictions",
                    "selector": "$steps.step_4.predictions",
                },
            ],
        }
    )

    # when
    result = construct_graph(deployment_spec=deployment_specs)

    # then
    assert (
        result.nodes["$inputs.image"]["definition"].name == "image"
    ), "Image node must be named correctly"
    assert (
        result.nodes["$inputs.confidence"]["definition"].name == "confidence"
    ), "Input confidence node must be named correctly"
    assert (
        result.nodes["$steps.step_1"]["definition"].name == "step_1"
    ), "Step 1 node must be named correctly"
    assert (
        result.nodes["$steps.step_2"]["definition"].name == "step_2"
    ), "Step 2 node must be named correctly"
    assert (
        result.nodes["$steps.step_3"]["definition"].name == "step_3"
    ), "Step 3 node must be named correctly"
    assert (
        result.nodes["$steps.step_4"]["definition"].name == "step_4"
    ), "Step 4 node must be named correctly"
    assert (
        result.nodes["$outputs.top_class"]["definition"].selector == "$steps.step_1.top"
    ), "Output must be installed correctly"
    assert (
        result.nodes["$outputs.step_3_predictions"]["definition"].selector
        == "$steps.step_3.predictions"
    ), "Output must be installed correctly"
    assert (
        result.nodes["$outputs.step_4_predictions"]["definition"].selector
        == "$steps.step_4.predictions"
    ), "Output must be installed correctly"
    assert result.has_edge(
        "$inputs.image", "$steps.step_1"
    ), "Image must be connected with step 1"
    assert result.has_edge(
        "$inputs.confidence", "$steps.step_1"
    ), "Confidence parameter must be connected with step 1"
    assert result.has_edge(
        "$inputs.image", "$steps.step_3"
    ), "Image must be connected with step 3"
    assert result.has_edge(
        "$inputs.image", "$steps.step_4"
    ), "Image must be connected with step 4"
    assert result.has_edge(
        "$steps.step_1", "$steps.step_2"
    ), "Object detection node must be connected with Condition step"
    assert result.has_edge(
        "$steps.step_2", "$steps.step_3"
    ), "Condition step must be connected with step 3"
    assert result.has_edge(
        "$steps.step_2", "$steps.step_4"
    ), "Condition step must be connected with step 4"
    assert result.has_edge(
        "$steps.step_1", "$outputs.top_class"
    ), "Step 1 must be connected to top_class output"
    assert result.has_edge(
        "$steps.step_3", "$outputs.step_3_predictions"
    ), "Step 3 must be connected to step_3_predictions output"
    assert result.has_edge(
        "$steps.step_4", "$outputs.step_4_predictions"
    ), "Step 4 must be connected to step_4_predictions output"
    assert len(result.edges) == 10, "10 edges in total should be created"
