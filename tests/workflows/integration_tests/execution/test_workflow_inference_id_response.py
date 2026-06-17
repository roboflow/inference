import numpy as np
import pytest

from inference.core.env import (
    ENABLE_TENSOR_DATA_REPRESENTATION,
    WORKFLOWS_MAX_CONCURRENT_STEPS,
)
from inference.core.managers.base import ModelManager
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    INFERENCE_ID_KEY,
)
from inference.core.workflows.execution_engine.core import ExecutionEngine
from inference_models.models.base.classification import ClassificationPrediction
from inference_models.models.base.object_detection import Detections as NativeDetections

# Under ENABLE_TENSOR_DATA_REPRESENTATION the detection/classification blocks emit native
# inference_models dataclasses (Detections / ClassificationPrediction) instead of
# sv.Detections / sv-shaped dicts, so inference_id moves to image-level metadata. The
# numpy-shaped assertions below run only with the flag off; an equivalent *_tensor_native
# test asserts the same facts against the native carrier with the flag on.
_NUMPY_ONLY = pytest.mark.skipif(
    ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="sv.Detections / sv-shaped output; native under "
    "ENABLE_TENSOR_DATA_REPRESENTATION — see the *_tensor_native parity test",
)
_TENSOR_ONLY = pytest.mark.skipif(
    not ENABLE_TENSOR_DATA_REPRESENTATION,
    reason="tensor-native variant; runs only with ENABLE_TENSOR_DATA_REPRESENTATION=True",
)

DETECTION_PLUS_CLASSIFICATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
        {
            "type": "Crop",
            "name": "cropping",
            "image": "$inputs.image",
            "predictions": "$steps.general_detection.predictions",
        },
        {
            "type": "ClassificationModel",
            "name": "breds_classification",
            "image": "$steps.cropping.crops",
            "model_id": "dog-breed-xpaq6/1",
            "confidence": 0.09,
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.breds_classification.predictions",
        },
    ],
}

OBJECT_DETECTION_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "ObjectDetectionModel",
            "name": "general_detection",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
            "class_filter": ["dog"],
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "coordinates_system": "own",
            "selector": "$steps.general_detection.predictions",
        }
    ],
}

INSTANCE_SEGMENTATION_WORKFLOW = {
    "version": "1.0",
    "inputs": [{"type": "WorkflowImage", "name": "image"}],
    "steps": [
        {
            "type": "InstanceSegmentationModel",
            "name": "instance_segmentation",
            "image": "$inputs.image",
            "model_id": "yolov8n-640",
        },
    ],
    "outputs": [
        {
            "type": "JsonField",
            "name": "predictions",
            "selector": "$steps.instance_segmentation.*",
        }
    ],
}


@_NUMPY_ONLY
@pytest.mark.workflows
def test_detection_plus_classification_workflow_with_inference_id(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTION_PLUS_CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert (
        len(result[0]["predictions"]) == 2
    ), "Expected 2 dogs crops on input image, hence 2 nested classification results"

    for prediction in result[0]["predictions"]:
        assert "inference_id" in prediction, "Expected inference_id in each prediction"
        assert prediction["inference_id"] is not None, "Expected non-null inference_id"

    assert [result[0]["predictions"][0]["top"], result[0]["predictions"][1]["top"]] == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected predictions to be as measured in reference run"


@_TENSOR_ONLY
@pytest.mark.workflows
def test_detection_plus_classification_workflow_with_inference_id_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=DETECTION_PLUS_CLASSIFICATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert (
        len(result[0]["predictions"]) == 2
    ), "Expected 2 dogs crops on input image, hence 2 nested classification results"

    # Under the flag each nested classification result is a native
    # ClassificationPrediction dataclass (not iterable). inference_id lives on
    # image-level metadata at images_metadata[0][INFERENCE_ID_KEY]; the top class is
    # resolved via images_metadata[0][CLASS_NAMES_KEY][top class_id].
    tops = []
    for prediction in result[0]["predictions"]:
        assert isinstance(
            prediction, ClassificationPrediction
        ), "Expected native ClassificationPrediction under the flag"
        meta = prediction.images_metadata[0]
        assert INFERENCE_ID_KEY in meta, "Expected inference_id in image metadata"
        assert meta[INFERENCE_ID_KEY] is not None, "Expected non-null inference_id"
        top_class_id = int(prediction.class_id.reshape(-1)[0])
        tops.append(meta[CLASS_NAMES_KEY][top_class_id])

    assert tops == [
        "116.Parson_russell_terrier",
        "131.Wirehaired_pointing_griffon",
    ], "Expected predictions to be as measured in reference run"


@_NUMPY_ONLY
@pytest.mark.workflows
def test_object_detection_workflow_with_inference_id(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert len(result[0]["predictions"]) == 2, "Expected 2 predictions"
    assert (
        result[0]["predictions"][0]["inference_id"] is not None
    ), "Expected non-null inference_id"
    assert (
        result[0]["predictions"][1]["inference_id"] is not None
    ), "Expected non-null inference_id"


@_TENSOR_ONLY
@pytest.mark.workflows
def test_object_detection_workflow_with_inference_id_tensor_native(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=OBJECT_DETECTION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    # Under the flag predictions is a native Detections carrier (no __getitem__).
    # inference_id is image-level metadata shared by both boxes, not per-box.
    dets = result[0]["predictions"]
    assert isinstance(
        dets, NativeDetections
    ), "Expected native inference_models.Detections under the flag"
    assert dets.xyxy.shape[0] == 2, "Expected 2 predictions"
    assert (
        dets.image_metadata[INFERENCE_ID_KEY] is not None
    ), "Expected non-null inference_id"


@pytest.mark.workflows
def test_instance_segmentation_workflow_with_inference_id(
    model_manager: ModelManager,
    dogs_image: np.ndarray,
    roboflow_api_key: str,
) -> None:
    # given
    workflow_init_parameters = {
        "workflows_core.model_manager": model_manager,
        "workflows_core.api_key": roboflow_api_key,
        "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
    }
    execution_engine = ExecutionEngine.init(
        workflow_definition=INSTANCE_SEGMENTATION_WORKFLOW,
        init_parameters=workflow_init_parameters,
        max_concurrent_steps=WORKFLOWS_MAX_CONCURRENT_STEPS,
    )

    # when
    result = execution_engine.run(
        runtime_parameters={
            "image": dogs_image,
        }
    )

    # then
    assert len(result[0]["predictions"]) == 2, "Expected 2 predictions"
    assert (
        result[0]["predictions"].get("inference_id") is not None
    ), "Expected non-null inference_id"
