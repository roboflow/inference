import base64
import io

import numpy as np
import supervision as sv
from PIL import Image


def assert_localized_predictions_match(
    result_prediction: dict,
    reference_prediction: dict,
    box_pixel_tolerance: float = 1,
    box_confidence_tolerance: float = 1e-3,
    mask_iou_threshold: float = 0.999,
    keypoint_pixel_tolerance: float = 1,
    keypoint_confidence_tolerance: float = 5e-2,
) -> None:
    sv_result_prediction = sv.Detections.from_inference(result_prediction)
    sv_reference_prediction = sv.Detections.from_inference(reference_prediction)

    # the sv prediction objects have attributes in batch format, so we run batch-based comparisons
    # NOTE: this requires that the detections are in the same order in both predictions
    # let's leave addressing that to a future issue .. by default these are likely sorted by confidence which imposes a meaningful order
    # in that, if after sorting by confidence the predictions are not ordered the same, likely they wouldn't pass this assertion anyway
    # the rigid assumption there is that the smallest gap between confidences is higher than our similarity threshold

    assert len(sv_result_prediction) == len(
        sv_reference_prediction
    ), "Predictions must have the same number of detections"

    assert np.allclose(
        sv_result_prediction.xyxy,
        sv_reference_prediction.xyxy,
        atol=box_pixel_tolerance,
    ), (
        f"Bounding boxes must match with a tolerance of {box_pixel_tolerance} pixels, "
        f"got {sv_result_prediction.xyxy} and {sv_reference_prediction.xyxy}"
    )

    if sv_reference_prediction.confidence is not None:
        assert np.allclose(
            sv_result_prediction.confidence,
            sv_reference_prediction.confidence,
            atol=box_confidence_tolerance,
        ), (
            f"Confidence must match with a tolerance of {box_confidence_tolerance}, "
            f"got {sv_result_prediction.confidence} and {sv_reference_prediction.confidence}"
        )

    if sv_reference_prediction.class_id is not None:
        assert np.array_equal(
            sv_result_prediction.class_id, sv_reference_prediction.class_id
        ), f"Class IDs must match, got {sv_result_prediction.class_id} and {sv_reference_prediction.class_id}"

    # now for keypoint and mask specific assertions

    if sv_reference_prediction.mask is not None:
        assert (
            sv_result_prediction.mask is not None
        ), "Mask must be present for instance segmentation predictions"
        iou = np.sum(
            sv_result_prediction.mask & sv_reference_prediction.mask, axis=(1, 2)
        ) / np.sum(
            sv_result_prediction.mask | sv_reference_prediction.mask, axis=(1, 2)
        )
        assert np.all(
            iou > mask_iou_threshold
        ), f"Mask IOU must be greater than {mask_iou_threshold} for all predictions, got {iou}"

    if all("keypoints" not in p for p in reference_prediction["predictions"]):
        return None
    # have to separately create a KeyPoints object to compare keypoints
    result_prediction_keypoints = sv.KeyPoints.from_inference(result_prediction)
    reference_prediction_keypoints = sv.KeyPoints.from_inference(reference_prediction)

    assert len(result_prediction_keypoints) == len(
        reference_prediction_keypoints
    ), "Keypoints must have the same number of keypoints"

    assert np.allclose(
        result_prediction_keypoints.xy,
        reference_prediction_keypoints.xy,
        atol=keypoint_pixel_tolerance,
    ), (
        f"Keypoints must match with a tolerance of {keypoint_pixel_tolerance} pixels, "
        f"got {result_prediction_keypoints.xy} and {reference_prediction_keypoints.xy}"
    )

    if result_prediction_keypoints.confidence is not None:
        # NOTE: this was not one of the original test cases, but likely useful so adding it
        assert np.allclose(
            result_prediction_keypoints.confidence,
            reference_prediction_keypoints.confidence,
            atol=keypoint_confidence_tolerance,
        ), (
            f"Keypoint confidence must match with a tolerance of {keypoint_confidence_tolerance}, "
            f"got {result_prediction_keypoints.confidence} and {reference_prediction_keypoints.confidence}"
        )

    if result_prediction_keypoints.class_id is not None:
        assert np.array_equal(
            result_prediction_keypoints.class_id,
            reference_prediction_keypoints.class_id,
        ), f"Keypoint class IDs must match, got {result_prediction_keypoints.class_id} and {reference_prediction_keypoints.class_id}"


def assert_classification_predictions_match(
    result_prediction: dict,
    reference_prediction: dict,
    confidence_tolerance: float = 1e-5,
) -> None:
    assert type(result_prediction) == type(
        reference_prediction
    ), "Predictions must be of the same type"
    assert len(result_prediction["predictions"]) == len(
        reference_prediction["predictions"]
    ), "Predictions must have the same number of predictions"
    if isinstance(reference_prediction["predictions"], dict):
        assert sorted(result_prediction["predicted_classes"]) == sorted(
            reference_prediction["predicted_classes"]
        ), f"Predicted classes must match, got {result_prediction['predicted_classes']} and {result_prediction['predicted_classes']}"
    else:
        assert (
            result_prediction["top"] == reference_prediction["top"]
        ), f"Top prediction must match, got {result_prediction['top']} and {reference_prediction['top']}"
        assert np.allclose(
            result_prediction["confidence"],
            reference_prediction["confidence"],
            atol=confidence_tolerance,
        ), (
            f"Confidences must match with a tolerance of {confidence_tolerance}, "
            f"got {result_prediction['confidence']} and {reference_prediction['confidence']}"
        )


def _decode_b64_mask(b64_str: str) -> np.ndarray:
    """Decode a base64-encoded PNG string into a uint8 numpy array."""
    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw))
    return np.array(img, dtype=np.uint8)


def assert_semantic_segmentation_predictions_match(
    result_prediction: dict,
    reference_prediction: dict,
    segmentation_mask_iou_threshold: float = 0.999,
    confidence_mask_pixel_tolerance: int = 1,
) -> None:
    """Assert that two SemanticSegmentationInferenceResponse dicts match.

    Compares:
    - class_map equality
    - segmentation_mask pixel-level IoU (per class)
    - confidence_mask pixel-level closeness
    - image dimensions
    """
    result_preds = result_prediction["predictions"]
    reference_preds = reference_prediction["predictions"]

    # class_map must match exactly
    assert result_preds["class_map"] == reference_preds["class_map"], (
        f"Class maps must match, got {result_preds['class_map']} "
        f"and {reference_preds['class_map']}"
    )

    # decode masks
    result_seg = _decode_b64_mask(result_preds["segmentation_mask"])
    reference_seg = _decode_b64_mask(reference_preds["segmentation_mask"])

    assert result_seg.shape == reference_seg.shape, (
        f"Segmentation mask shapes must match, "
        f"got {result_seg.shape} and {reference_seg.shape}"
    )

    # compute IoU over the full segmentation mask (treating matching pixels as intersection)
    matching = result_seg == reference_seg
    iou = np.sum(matching) / matching.size
    assert iou >= segmentation_mask_iou_threshold, (
        f"Segmentation mask IoU must be >= {segmentation_mask_iou_threshold}, got {iou}"
    )

    # confidence masks should be close
    result_conf = _decode_b64_mask(result_preds["confidence_mask"])
    reference_conf = _decode_b64_mask(reference_preds["confidence_mask"])

    assert result_conf.shape == reference_conf.shape, (
        f"Confidence mask shapes must match, "
        f"got {result_conf.shape} and {reference_conf.shape}"
    )

    assert np.allclose(
        result_conf.astype(np.int16),
        reference_conf.astype(np.int16),
        atol=confidence_mask_pixel_tolerance,
    ), (
        f"Confidence masks must match within tolerance of "
        f"{confidence_mask_pixel_tolerance} intensity levels"
    )

    # image dimensions
    assert result_prediction["image"] == reference_prediction["image"], (
        f"Image dimensions must match, got {result_prediction['image']} "
        f"and {reference_prediction['image']}"
    )
