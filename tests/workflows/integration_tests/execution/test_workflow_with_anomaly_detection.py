import numpy as np
import pytest

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    USE_INFERENCE_MODELS,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.serializers_tensor import (
    serialise_native_classification,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from tests.workflows.integration_tests.execution.workflows_gallery_collector.decorators import (
    add_to_workflows_gallery,
)

pytestmark = pytest.mark.skipif(
    not USE_INFERENCE_MODELS,
    reason="PatchCore and FoundAD load only under the inference_models runtime",
)

# Public models of the anomaly reference project, trained on the same data.
FOUNDAD_MODEL_ID = "paul-guerrie-tang1/anomaly-det-algbq-1-foundad-t1"
PATCHCORE_MODEL_ID = "paul-guerrie-tang1/anomaly-det-algbq-1-patchcore-t2"

WORKFLOW_DEFINITION = {
    "version": "1.0",
    "inputs": [
        {"type": "WorkflowImage", "name": "image"},
        {
            "type": "WorkflowParameter",
            "name": "model_id",
            "default_value": FOUNDAD_MODEL_ID,
        },
        {"type": "WorkflowParameter", "name": "confidence", "default_value": 0.5},
        {"type": "WorkflowParameter", "name": "anomaly_threshold"},
    ],
    "steps": [
        {
            "type": "roboflow_core/anomaly_detection_model@v1",
            "name": "model",
            "images": "$inputs.image",
            "model_id": "$inputs.model_id",
            "confidence": "$inputs.confidence",
            "anomaly_threshold": "$inputs.anomaly_threshold",
        },
        {
            "type": "roboflow_core/anomaly_heatmap_visualization@v1",
            "name": "heatmap",
            "image": "$inputs.image",
            "anomaly_map": "$steps.model.anomaly_map",
            "anomaly_threshold": "$steps.model.anomaly_decision_threshold",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.model.predictions",
        },
        {
            "type": "JsonField",
            "name": "is_anomalous",
            "selector": "$steps.model.is_anomalous",
        },
        {
            "type": "JsonField",
            "name": "anomaly_score",
            "selector": "$steps.model.anomaly_score",
        },
        {
            "type": "JsonField",
            "name": "anomaly_threshold",
            "selector": "$steps.model.anomaly_threshold",
        },
        {
            "type": "JsonField",
            "name": "anomaly_decision_threshold",
            "selector": "$steps.model.anomaly_decision_threshold",
        },
        {
            "type": "JsonField",
            "name": "heatmap",
            "selector": "$steps.heatmap.image",
        },
    ],
}


def _engine(model_manager: ModelManager, roboflow_api_key: str) -> ExecutionEngine:
    return ExecutionEngine.init(
        workflow_definition=WORKFLOW_DEFINITION,
        init_parameters={
            "workflows_core.model_manager": model_manager,
            "workflows_core.api_key": roboflow_api_key,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )


def _decision(result: dict):
    predictions = result["predictions"]
    if ENABLE_TENSOR_DATA_REPRESENTATION:
        # The flag ships a native ClassificationPrediction; the serializer restores
        # the numpy dict shape.
        predictions = serialise_native_classification(prediction=predictions)
    return result["is_anomalous"], predictions["top"]


@add_to_workflows_gallery(
    category="Workflows with visualization blocks",
    use_case_title="Workflow with anomaly detection and its heatmap",
    workflow_name_in_app="anomaly-detection-heatmap",
    use_case_description="""
This workflow runs an anomaly detection model (FoundAD or PatchCore) and overlays its
anomaly map on the image. The heatmap is anchored at the block's decision threshold,
so the middle of the colour ramp marks the anomalous verdict at any confidence.
    """,
    workflow_definition=WORKFLOW_DEFINITION,
)
@pytest.mark.parametrize("model_id", [FOUNDAD_MODEL_ID, PATCHCORE_MODEL_ID])
def test_anomaly_detection_workflow_decides_and_draws_the_heatmap(
    model_manager: ModelManager,
    anomalous_image: np.ndarray,
    normal_image: np.ndarray,
    roboflow_api_key: str,
    model_id: str,
) -> None:
    # given
    execution_engine = _engine(model_manager, roboflow_api_key)

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": [anomalous_image, normal_image],
            "model_id": model_id,
        }
    )

    # then
    assert len(result) == 2, "Expected one output per input image"
    assert _decision(result[0]) == (True, "anomalous")
    assert _decision(result[1]) == (False, "normal")
    for image_result, image in zip(result, (anomalous_image, normal_image)):
        # At the default confidence the decision sits at the saved threshold.
        assert image_result["anomaly_decision_threshold"] == pytest.approx(
            image_result["anomaly_threshold"]
        )
        assert image_result["heatmap"].numpy_image.shape == image.shape
        assert not np.array_equal(image_result["heatmap"].numpy_image, image)


@pytest.mark.parametrize("model_id", [FOUNDAD_MODEL_ID, PATCHCORE_MODEL_ID])
def test_anomaly_detection_workflow_confidence_moves_the_decision(
    model_manager: ModelManager,
    anomalous_image: np.ndarray,
    normal_image: np.ndarray,
    roboflow_api_key: str,
    model_id: str,
) -> None:
    # given
    execution_engine = _engine(model_manager, roboflow_api_key)

    # when: every anomalous confidence reaches 0, none reaches 1
    lenient = execution_engine.run(
        runtime_parameters={
            "image": normal_image,
            "model_id": model_id,
            "confidence": 0.0,
        }
    )
    strict = execution_engine.run(
        runtime_parameters={
            "image": anomalous_image,
            "model_id": model_id,
            "confidence": 1.0,
        }
    )

    # then
    assert _decision(lenient[0]) == (True, "anomalous")
    assert _decision(strict[0]) == (False, "normal")
    assert lenient[0]["anomaly_decision_threshold"] < lenient[0]["anomaly_threshold"]
    assert strict[0]["anomaly_decision_threshold"] > strict[0]["anomaly_threshold"]
    # Past a decision threshold below zero the whole map is anomalous evidence.
    assert lenient[0]["heatmap"].numpy_image.shape == normal_image.shape


@pytest.mark.parametrize("model_id", [FOUNDAD_MODEL_ID, PATCHCORE_MODEL_ID])
def test_anomaly_detection_workflow_threshold_override_composes_with_confidence(
    model_manager: ModelManager,
    anomalous_image: np.ndarray,
    roboflow_api_key: str,
    model_id: str,
) -> None:
    # given
    execution_engine = _engine(model_manager, roboflow_api_key)
    baseline = execution_engine.run(
        runtime_parameters={"image": anomalous_image, "model_id": model_id}
    )[0]
    above_the_score = baseline["anomaly_score"] * 2

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": anomalous_image,
            "model_id": model_id,
            "anomaly_threshold": above_the_score,
        }
    )[0]

    # then
    assert _decision(result) == (False, "normal")
    assert result["anomaly_threshold"] == above_the_score
    assert result["anomaly_decision_threshold"] == pytest.approx(above_the_score)
