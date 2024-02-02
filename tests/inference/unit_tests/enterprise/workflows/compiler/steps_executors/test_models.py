from unittest import mock

from inference.enterprise.workflows.complier.steps_executors import models
from inference.enterprise.workflows.complier.steps_executors.models import (
    construct_http_client_configuration_for_classification_step,
    construct_http_client_configuration_for_detection_step,
    construct_http_client_configuration_for_keypoints_detection_step,
    construct_http_client_configuration_for_segmentation_step,
    resolve_model_api_url,
)
from inference.enterprise.workflows.entities.steps import (
    ClassificationModel,
    ClipComparison,
    InstanceSegmentationModel,
    KeypointsDetectionModel,
    MultiLabelClassificationModel,
    ObjectDetectionModel,
    OCRModel,
)


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "self-hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_self_hosted_api_is_chosen() -> None:
    # given
    some_step = ClassificationModel(
        type="ClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "http://127.0.0.1:9001"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_classification_model() -> (
    None
):
    # given
    some_step = ClassificationModel(
        type="ClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://classify.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_multi_label_classification_model() -> (
    None
):
    # given
    some_step = MultiLabelClassificationModel(
        type="MultiLabelClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://classify.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_detection_model() -> None:
    # given
    some_step = ObjectDetectionModel(
        type="ObjectDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://detect.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_keypoints_model() -> None:
    # given
    some_step = KeypointsDetectionModel(
        type="KeypointsDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://detect.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_segmentation_model() -> (
    None
):
    # given
    some_step = InstanceSegmentationModel(
        type="InstanceSegmentationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://outline.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_ocr_model() -> None:
    # given
    some_step = OCRModel(
        type="OCRModel",
        name="some",
        image="$inputs.image",
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://infer.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_API_TARGET", "hosted")
@mock.patch.object(models, "LOCAL_INFERENCE_API_URL", "http://127.0.0.1:9001")
def test_resolve_model_api_url_when_hosted_api_is_chosen_for_clip_model() -> None:
    # given
    some_step = ClipComparison(
        type="ClipComparison", name="some", image="$inputs.image", text=["a", "b"]
    )

    # when
    result = resolve_model_api_url(step=some_step)

    # then
    assert result == "https://infer.roboflow.com"


@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", 4)
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
def test_construct_http_client_configuration_for_classification_step() -> None:
    # given
    some_step = ClassificationModel(
        type="ClassificationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
        confidence="$inputs.confidence",
    )
    runtime_parameters = {
        "confidence": 0.7,
    }

    # when
    result = construct_http_client_configuration_for_classification_step(
        step=some_step,
        runtime_parameters=runtime_parameters,
        outputs_lookup={},
    )

    # then
    assert (
        result.confidence_threshold == 0.7
    ), "Confidence threshold must be resolved from runtime params"
    assert result.disable_active_learning is False, "Flag must have default value"
    assert result.max_batch_size == 4, "value must match env config"
    assert result.max_concurrent_requests == 2, "value must match env config"


@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", 4)
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
def test_construct_http_client_configuration_for_detection_step_step() -> None:
    # given
    some_step = ObjectDetectionModel(
        type="ObjectDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
        confidence="$inputs.confidence",
        iou_threshold="$inputs.iou_threshold",
    )
    runtime_parameters = {"confidence": 0.7, "iou_threshold": 0.1}

    # when
    result = construct_http_client_configuration_for_detection_step(
        step=some_step,
        runtime_parameters=runtime_parameters,
        outputs_lookup={},
    )

    # then
    assert (
        result.confidence_threshold == 0.7
    ), "Confidence threshold must be resolved from runtime params"
    assert (
        result.iou_threshold == 0.1
    ), "Iou threshold must be resolved from runtime params"
    assert result.disable_active_learning is False, "Flag must have default value"
    assert result.max_batch_size == 4, "value must match env config"
    assert result.max_concurrent_requests == 2, "value must match env config"


@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", 4)
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
def test_construct_http_client_configuration_for_segmentation_step() -> None:
    # given
    some_step = InstanceSegmentationModel(
        type="InstanceSegmentationModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
        confidence="$inputs.confidence",
        iou_threshold="$inputs.iou_threshold",
    )
    runtime_parameters = {"confidence": 0.7, "iou_threshold": 0.1}

    # when
    result = construct_http_client_configuration_for_segmentation_step(
        step=some_step,
        runtime_parameters=runtime_parameters,
        outputs_lookup={},
    )

    # then
    assert (
        result.confidence_threshold == 0.7
    ), "Confidence threshold must be resolved from runtime params"
    assert (
        result.iou_threshold == 0.1
    ), "Iou threshold must be resolved from runtime params"
    assert result.disable_active_learning is False, "Flag must have default value"
    assert result.max_batch_size == 4, "value must match env config"
    assert result.max_concurrent_requests == 2, "value must match env config"


@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE", 4)
@mock.patch.object(models, "WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS", 2)
def test_construct_http_client_configuration_for_keypoints_detection_step_step() -> (
    None
):
    # given
    some_step = KeypointsDetectionModel(
        type="KeypointsDetectionModel",
        name="some",
        image="$inputs.image",
        model_id="some/1",
        confidence="$inputs.confidence",
        iou_threshold="$inputs.iou_threshold",
        keypoint_confidence=1.0,
    )
    runtime_parameters = {"confidence": 0.7, "iou_threshold": 0.1}

    # when
    result = construct_http_client_configuration_for_keypoints_detection_step(
        step=some_step,
        runtime_parameters=runtime_parameters,
        outputs_lookup={},
    )

    # then
    assert (
        result.confidence_threshold == 0.7
    ), "Confidence threshold must be resolved from runtime params"
    assert (
        result.iou_threshold == 0.1
    ), "Iou threshold must be resolved from runtime params"
    assert (
        result.keypoint_confidence_threshold == 1.0
    ), "keypoints threshold must be as defined in step initialisation"
    assert result.disable_active_learning is False, "Flag must have default value"
    assert result.max_batch_size == 4, "value must match env config"
    assert result.max_concurrent_requests == 2, "value must match env config"
