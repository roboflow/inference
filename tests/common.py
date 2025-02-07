import supervision as sv
from typing import Union
import numpy as np
from inference.core.entities.responses.inference import (
    ClassificationInferenceResponse,
    InstanceSegmentationInferenceResponse,
    KeypointsDetectionInferenceResponse,
    ObjectDetectionInferenceResponse,
)


def assert_localized_predictions_match(
        prediction_1: Union[ObjectDetectionInferenceResponse, InstanceSegmentationInferenceResponse, KeypointsDetectionInferenceResponse],
        prediction_2: Union[ObjectDetectionInferenceResponse, InstanceSegmentationInferenceResponse, KeypointsDetectionInferenceResponse],
        box_pixel_tolerance: float = 1,
        box_confidence_tolerance: float = 1e-3,
        mask_iou_threshold: float = 0.999,
        keypoint_pixel_tolerance: float = 1,
        keypoint_confidence_tolerance: float = 5e-2,
) -> None:
    # we rely on supervision here because it automatically converts polygons to masks and handles batching nicely
    assert type(prediction_1) == type(prediction_2), "Predictions must be of the same type"

    sv_prediction_1 = sv.Detections.from_inference(prediction_1)
    sv_prediction_2 = sv.Detections.from_inference(prediction_2)

    # the sv prediction objects have attributes in batch format, so we run batch-based comparisons
    # NOTE: this requires that the detections are in the same order in both predictions
    # let's leave addressing that to a future issue .. by default these are likely sorted by confidence which imposes a meaningful order
    # in that, if after sorting by confidence the predictions are not ordered the same, likely they wouldn't pass this assertion anyway
    # the rigid assumption there is that the smallest gap between confidences is higher than our similarity threshold

    assert len(sv_prediction_1) == len(sv_prediction_2), "Predictions must have the same number of detections"

    assert np.allclose(sv_prediction_1.xyxy, sv_prediction_2.xyxy, atol=box_pixel_tolerance), f"Bounding boxes must match with a tolerance of {box_pixel_tolerance} pixels"

    if sv_prediction_1.confidence is not None:
        assert np.allclose(sv_prediction_1.confidence, sv_prediction_2.confidence, atol=box_confidence_tolerance), f"Confidence must match with a tolerance of {box_confidence_tolerance}"

    if sv_prediction_1.class_id is not None:
        assert np.array_equal(sv_prediction_1.class_id, sv_prediction_2.class_id), "Class IDs must match"
    
    # now for keypoint and mask specific assertions

    if isinstance(prediction_1, InstanceSegmentationInferenceResponse):
        assert sv_prediction_1.mask is not None, "Mask must be present for instance segmentation predictions"
        iou = np.sum(sv_prediction_1.mask & sv_prediction_2.mask, axis=(1, 2)) / np.sum(sv_prediction_1.mask | sv_prediction_2.mask, axis=(1, 2))
        assert np.all(iou > mask_iou_threshold), f"Mask IOU must be greater than {mask_iou_threshold} for all predictions"

    if isinstance(prediction_1, KeypointsDetectionInferenceResponse):
        # have to separately create a KeyPoints object to compare keypoints
        sv_keypoints_1 = sv.KeyPoints.from_inference(prediction_1)
        sv_keypoints_2 = sv.KeyPoints.from_inference(prediction_2)

        assert len(sv_keypoints_1) == len(sv_keypoints_2), "Keypoints must have the same number of keypoints"

        assert np.allclose(sv_keypoints_1.xy, sv_keypoints_2.xy, atol=keypoint_pixel_tolerance), f"Keypoints must match with a tolerance of {keypoint_pixel_tolerance} pixels"

        if sv_keypoints_1.confidence is not None:
            # NOTE: this was not one of the original test cases, but likely useful so adding it
            assert np.allclose(sv_keypoints_1.confidence, sv_keypoints_2.confidence, atol=keypoint_confidence_tolerance), f"Keypoint confidence must match with a tolerance of {keypoint_confidence_tolerance}"
        
        if sv_keypoints_1.class_id is not None:
            assert np.array_equal(sv_keypoints_1.class_id, sv_keypoints_2.class_id), "Keypoint class IDs must match"


def assert_classification_predictions_match(
    prediction_1: ClassificationInferenceResponse,
    prediction_2: ClassificationInferenceResponse,
    confidence_tolerance: float = 1e-5,
) -> None:
    assert len(prediction_1.predictions) == len(prediction_2.predictions), "Predictions must have the same number of predictions"
    assert prediction_1.top == prediction_2.top, "Top class must match"
    assert np.allclose(prediction_1.confidence, prediction_2.confidence, atol=confidence_tolerance), f"Confidence must match with a tolerance of {confidence_tolerance}"
