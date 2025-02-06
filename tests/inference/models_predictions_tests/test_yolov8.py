import numpy as np
import pytest
import json
import supervision as sv

from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    ObjectDetectionInferenceResponse,
)
from inference.core.env import MAX_BATCH_SIZE
from inference.models import (
    YOLOv8Classification,
    YOLOv8InstanceSegmentation,
    YOLOv8KeypointsDetection,
    YOLOv8ObjectDetection,
)


@pytest.mark.slow
def test_yolov8_classification_single_image_inference(
    yolov8_cls_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = YOLOv8Classification(model_id=yolov8_cls_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_yolov8_classification_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_yolov8_classification_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov8_cls_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv8Classification(model_id=yolov8_cls_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov8_classification_prediction_matches_reference(prediction=prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov8_classification_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov8_cls_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv8Classification(model_id=yolov8_cls_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov8_classification_prediction_matches_reference(prediction=prediction)


def assert_yolov8_classification_prediction_matches_reference(
    prediction: ClassificationInferenceResponse,
) -> None:
    assert (
        len(prediction.predictions) == 1000
    ), "Example model is expected to predict across 1000 classes"
    assert prediction.top == "golden_retriever", "Golden retriever should be predicted"
    assert (
        abs(prediction.confidence - 0.0018) < 1e-5
    ), "Confidence while test creation was 0.0018"


@pytest.mark.slow
def test_yolov8_detection_single_image_inference(
    yolov8_det_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = YOLOv8ObjectDetection(model_id=yolov8_det_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    print(result[0])
    assert_yolov8_detection_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_yolov8_detection_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov8_det_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv8ObjectDetection(model_id=yolov8_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov8_detection_prediction_matches_reference(prediction=prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov8_detection_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov8_det_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv8ObjectDetection(model_id=yolov8_det_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov8_detection_prediction_matches_reference(prediction=prediction)


def assert_yolov8_detection_prediction_matches_reference(
    prediction: ObjectDetectionInferenceResponse,
) -> None:
    assert (
        len(prediction.predictions) == 1
    ), "Example model is expected to predict 1 bbox, as this is the result obtained while test creation"
    assert (
        prediction.predictions[0].class_name == "dog"
    ), "Dog class was predicted by exported model"
    assert (
        abs(prediction.predictions[0].confidence - 0.892430) < 1e-3
    ), "Confidence while test creation was 0.892430"
    xywh = [
        prediction.predictions[0].x,
        prediction.predictions[0].y,
        prediction.predictions[0].width,
        prediction.predictions[0].height,
    ]
    assert np.allclose(
        xywh, [360.0, 215.5, 558.0, 411.0], atol=1.0
    ), "while test creation, box coordinates was [360.0, 215.5, 558.0, 411.0]"


@pytest.mark.slow
def test_yolov8_segmentation_single_image_inference(
    yolov8_seg_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    model = YOLOv8InstanceSegmentation(model_id=yolov8_seg_model, api_key="DUMMY")

    # when
    result = model.infer(example_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_yolov8_segmentation_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_yolov8_segmentation_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov8_seg_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv8InstanceSegmentation(model_id=yolov8_seg_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov8_segmentation_prediction_matches_reference(prediction=prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov8_segmentation_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov8_seg_model: str,
    example_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv8InstanceSegmentation(model_id=yolov8_seg_model, api_key="DUMMY")

    # when
    result = model.infer([example_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov8_segmentation_prediction_matches_reference(prediction=prediction)


def assert_yolov8_segmentation_prediction_matches_reference(
    prediction: InstanceSegmentationInferenceResponse,
) -> None:
    assert (
        len(prediction.predictions) == 1
    ), "Example model is expected to predict 1 instance, as this is the result obtained while test creation"
    assert (
        prediction.predictions[0].class_name == "dog"
    ), "Dog class was predicted by exported model"
    assert (
        abs(prediction.predictions[0].confidence - 0.9011) < 1e-3
    ), "Confidence while test creation was 0.9011"
    xywh = [
        prediction.predictions[0].x,
        prediction.predictions[0].y,
        prediction.predictions[0].width,
        prediction.predictions[0].height,
    ]
    assert np.allclose(
        xywh, [343.0, 214.5, 584.0, 417.0], atol=1.0
    ), "while test creation, box coordinates was [343.0, 214.5, 584.0, 417.0]"
    
    
    sv_prediction = sv.Detections.from_inference(prediction)
    
    
    print(type(json.loads(prediction.model_dump_json())))
    # with open("yolov8_reference_prediction.json", "w") as f:
    #     # json.dump(json.loads(prediction.model_dump_json())['predictions'][0], f)
    #     json.dump(prediction.model_dump(by_alias=True), f)
    with open("yolov8_reference_prediction.json", "r") as f:
        reference_prediction = InstanceSegmentationInferenceResponse.model_validate(json.load(f))
    print(reference_prediction)
    print(type(reference_prediction))
    reference_prediction_sv = sv.Detections.from_inference(reference_prediction)
    print(reference_prediction_sv.mask)
    prediction_sv = sv.Detections.from_inference(prediction)
    print(prediction_sv.mask)

    reference_mask_np = np.array(reference_prediction_sv.mask)
    mask_np = np.array(prediction_sv.mask)
    print(reference_mask_np.shape)
    print(mask_np.shape)

    iou = np.sum(reference_mask_np & mask_np) / np.sum(reference_mask_np | mask_np)
    print(iou)

    assert iou > 0.99, "IOU is too low"
    # assert (
    #     len(prediction.predictions[0].points) == 673
    # ), "while test creation, mask had 673 points"


@pytest.mark.slow
def test_yolov8_pose_single_image_inference(
    yolov8_pose_model: str,
    person_image: np.ndarray,
) -> None:
    # given
    model = YOLOv8KeypointsDetection(model_id=yolov8_pose_model, api_key="DUMMY")

    # when
    result = model.infer(person_image)

    # then
    assert len(result) == 1, "Batch size=1 hence 1 result expected"
    assert_yolov8_pose_prediction_matches_reference(prediction=result[0])


@pytest.mark.slow
def test_yolov8_pose_batch_inference_when_batch_size_smaller_than_max_batch_size(
    yolov8_pose_model: str,
    person_image: np.ndarray,
) -> None:
    # given
    batch_size = min(4, MAX_BATCH_SIZE)
    model = YOLOv8KeypointsDetection(model_id=yolov8_pose_model, api_key="DUMMY")

    # when
    result = model.infer([person_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov8_pose_prediction_matches_reference(prediction=prediction)


@pytest.mark.slow
@pytest.mark.skipif(
    MAX_BATCH_SIZE > 8,
    reason="This test requires reasonably small MAX_BATCH_SIZE set via environment variable",
)
def test_yolov8_pose_batch_inference_when_batch_size_larger_than_max_batch_size(
    yolov8_pose_model: str,
    person_image: np.ndarray,
) -> None:
    # given
    batch_size = MAX_BATCH_SIZE + 2
    model = YOLOv8KeypointsDetection(model_id=yolov8_pose_model, api_key="DUMMY")

    # when
    result = model.infer([person_image] * batch_size)

    # then
    assert len(result) == batch_size, "Number of results must match batch size"
    for prediction in result:
        assert_yolov8_pose_prediction_matches_reference(prediction=prediction)


def assert_yolov8_pose_prediction_matches_reference(
    prediction: KeypointsDetectionInferenceResponse,
) -> None:
    assert (
        len(prediction.predictions) == 1
    ), "Example model is expected to predict 1 instance, as this is the result obtained while test creation"
    assert (
        prediction.predictions[0].class_name == "person"
    ), "Person class was predicted by exported model"
    assert (
        abs(prediction.predictions[0].confidence - 0.88805) < 1e-3
    ), "Confidence while test creation was 0.88805"
    xywh = [
        prediction.predictions[0].x,
        prediction.predictions[0].y,
        prediction.predictions[0].width,
        prediction.predictions[0].height,
    ]
    assert np.allclose(
        xywh, [314.5, 268.5, 81.0, 215.0], atol=0.6
    ), "while test creation, box coordinates was [314.5, 268.5, 81.0, 215.0]"
    assert (
        len(prediction.predictions[0].keypoints) == 17
    ), "Reference model has 17 keypoints defined"
    assert (
        prediction.predictions[0].keypoints[0].class_name == "nose"
    ), "First keypoint was nose, when test was created"
    kp_xy = [
        prediction.predictions[0].keypoints[0].x,
        prediction.predictions[0].keypoints[0].y,
    ]
    assert np.allclose(
        kp_xy, [322.0, 182.0], atol=0.6
    ), "while test creation, nose keypoint was at [322.0, 182.0]"
