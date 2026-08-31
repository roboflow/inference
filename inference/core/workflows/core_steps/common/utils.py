import logging
import math
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import cv2
import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD, ORIENTED_BOX_COORDINATES

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.sam2 import Sam2InferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import ModelEndpointType
from inference.core.workflows.core_steps.common.keypoints import (
    KEYPOINT_PADDING_CLASS_NAME,
)
from inference.core.workflows.execution_engine.constants import (
    DETECTION_ID_KEY,
    HEIGHT_KEY,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    KEYPOINTS_CLASS_ID_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CLASS_NAME_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_CONFIDENCE_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS,
    KEYPOINTS_KEY_IN_INFERENCE_RESPONSE,
    KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    PARENT_COORDINATES_KEY,
    PARENT_DIMENSIONS_KEY,
    PARENT_ID_KEY,
    POLYGON_KEY_IN_SV_DETECTIONS,
    PREDICTION_TYPE_KEY,
    RLE_MASK_KEY_IN_INFERENCE_RESPONSE,
    RLE_MASK_KEY_IN_SV_DETECTIONS,
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
    SCALING_RELATIVE_TO_PARENT_KEY,
    SCALING_RELATIVE_TO_ROOT_PARENT_KEY,
    WIDTH_KEY,
    X_KEY,
    Y_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    OriginCoordinatesSystem,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.v1.executor.utils import (
    wrap_with_context_snapshot,
)
from inference.core.workflows.prototypes.block import BlockResult

T = TypeVar("T")


def load_core_model(
    model_manager: ModelManager,
    inference_request: Union[
        DoctrOCRInferenceRequest,
        EasyOCRInferenceRequest,
        ClipCompareRequest,
        YOLOWorldInferenceRequest,
        Sam2InferenceRequest,
    ],
    core_model: str,
) -> str:
    version_id_field = f"{core_model}_version_id"
    core_model_id = (
        f"{core_model}/{inference_request.__getattribute__(version_id_field)}"
    )
    model_manager.add_model(
        core_model_id,
        inference_request.api_key,
        endpoint_type=ModelEndpointType.CORE_MODEL,
    )
    return core_model_id


def attach_prediction_type_info(
    predictions: List[Dict[str, Any]],
    prediction_type: str,
    key: str = PREDICTION_TYPE_KEY,
) -> List[Dict[str, Any]]:
    for result in predictions:
        result[key] = prediction_type
    return predictions


def filter_out_invalid_polygons(predictions: List[dict]) -> List[dict]:
    return [
        d for d in predictions if "points" not in d or len(d.get("points", [])) >= 3
    ]


def _get_or_create_detection_id(prediction: dict) -> object:
    if DETECTION_ID_KEY in prediction:
        return prediction[DETECTION_ID_KEY]
    return str(uuid.uuid4())


def attach_prediction_type_info_to_sv_detections_batch(
    predictions: List[sv.Detections],
    prediction_type: str,
    key: str = PREDICTION_TYPE_KEY,
) -> List[sv.Detections]:
    for prediction in predictions:
        prediction[key] = np.array([prediction_type] * len(prediction))
    return predictions


def convert_inference_detections_batch_to_sv_detections(
    predictions: List[Dict[str, Union[List[Dict[str, Any]], Any]]],
    predictions_key: str = "predictions",
    image_key: str = "image",
) -> List[sv.Detections]:
    batch_of_detections: List[sv.Detections] = []
    for p in predictions:
        width, height = p[image_key][WIDTH_KEY], p[image_key][HEIGHT_KEY]
        detections = sv.Detections.from_inference(p)
        raw_predictions = p[predictions_key]
        if len(detections) != len(raw_predictions):
            raw_predictions = filter_out_invalid_polygons(predictions=raw_predictions)
        parent_ids = [d.get(PARENT_ID_KEY, "") for d in raw_predictions]
        detection_ids = [_get_or_create_detection_id(d) for d in raw_predictions]
        detections[DETECTION_ID_KEY] = np.array(detection_ids)
        detections[PARENT_ID_KEY] = np.array(parent_ids)
        detections[IMAGE_DIMENSIONS_KEY] = np.array([[height, width]] * len(detections))
        if INFERENCE_ID_KEY in p:
            detections[INFERENCE_ID_KEY] = np.array(
                [p[INFERENCE_ID_KEY]] * len(detections)
            )

        rle_masks = [
            d.get(RLE_MASK_KEY_IN_INFERENCE_RESPONSE) or d.get("rle")
            for d in raw_predictions
        ]
        if any(m is not None for m in rle_masks):
            detections.data[RLE_MASK_KEY_IN_SV_DETECTIONS] = np.array(
                rle_masks, dtype=object
            )
        batch_of_detections.append(detections)
    return batch_of_detections


def add_inference_keypoints_to_sv_detections(
    inference_prediction: List[dict],
    detections: sv.Detections,
) -> sv.Detections:
    if len(inference_prediction) != len(detections):
        raise ValueError(
            f"Detected missmatch in number of detections in sv.Detections instance ({len(detections)}) "
            f"and `inference` predictions ({len(inference_prediction)}) while attempting to add keypoints metadata."
        )
    keypoints_class_names = []
    keypoints_class_ids = []
    keypoints_confidences = []
    keypoints_xy = []
    for inference_detection in inference_prediction:
        keypoints = inference_detection.get(KEYPOINTS_KEY_IN_INFERENCE_RESPONSE, [])
        keypoints_class_names.append(
            [k[KEYPOINTS_CLASS_NAME_KEY_IN_INFERENCE_RESPONSE] for k in keypoints]
        )
        keypoints_class_ids.append(
            [k[KEYPOINTS_CLASS_ID_KEY_IN_INFERENCE_RESPONSE] for k in keypoints]
        )
        keypoints_confidences.append(
            [k[KEYPOINTS_CONFIDENCE_KEY_IN_INFERENCE_RESPONSE] for k in keypoints]
        )
        keypoints_xy.append([[k[X_KEY], k[Y_KEY]] for k in keypoints])
    # Pad to uniform length so arrays are proper N-d numpy arrays instead of
    # object-dtype ragged arrays. Object-dtype arrays break supervision's
    # is_data_equal (used in Detections indexing/comparison).
    max_kps = max((len(kp) for kp in keypoints_xy), default=0)
    n = len(inference_prediction)
    padded_xy = np.zeros((n, max_kps, 2), dtype=np.float32)
    padded_conf = np.zeros((n, max_kps), dtype=np.float32)
    padded_class_id = np.zeros((n, max_kps), dtype=int)
    # Padding slots carry the empty class name so downstream consumers can tell
    # them apart from real keypoints (see common/keypoints.py).
    padded_class_name = np.full((n, max_kps), KEYPOINT_PADDING_CLASS_NAME, dtype=object)
    for i in range(n):
        k = len(keypoints_xy[i])
        if k > 0:
            padded_xy[i, :k] = keypoints_xy[i]
            padded_conf[i, :k] = keypoints_confidences[i]
            padded_class_id[i, :k] = keypoints_class_ids[i]
            padded_class_name[i, :k] = keypoints_class_names[i]
    detections[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = padded_xy
    detections[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS] = padded_conf
    detections[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS] = padded_class_id
    detections[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS] = padded_class_name
    return detections


def attach_parents_coordinates_to_batch_of_sv_detections(
    predictions: List[sv.Detections],
    images: Iterable[WorkflowImageData],
) -> List[sv.Detections]:
    result = []
    for prediction, image in zip(predictions, images):
        result.append(
            attach_parents_coordinates_to_sv_detections(
                detections=prediction,
                image=image,
            )
        )
    return result


def empty_detections_with_image_metadata(
    image_height: int,
    image_width: int,
) -> sv.Detections:
    """Create a zero-row ``sv.Detections`` that carries image dimensions in
    ``metadata`` so the numpy serialiser can emit real ``image.width`` /
    ``image.height`` for empty VLM results — matching the tensor-native path.

    ``sv.Detections.data`` is per-row, so zero rows means the serialiser's
    per-row loop never executes and any keys stored there are invisible.
    ``metadata`` is a free-form dict on ``sv.Detections`` that survives zero
    rows and is read by the serialiser as a fallback when no per-row
    ``IMAGE_DIMENSIONS_KEY`` is found.
    """
    return sv.Detections(
        xyxy=np.empty((0, 4), dtype=np.float32),
        metadata={IMAGE_DIMENSIONS_KEY: [image_height, image_width]},
    )


def attach_parents_coordinates_to_sv_detections(
    detections: sv.Detections,
    image: WorkflowImageData,
) -> sv.Detections:
    detections = attach_parent_coordinates_to_detections(
        detections=detections,
        parent_metadata=image.workflow_root_ancestor_metadata,
        parent_id_key=ROOT_PARENT_ID_KEY,
        coordinates_key=ROOT_PARENT_COORDINATES_KEY,
        dimensions_key=ROOT_PARENT_DIMENSIONS_KEY,
    )
    return attach_parent_coordinates_to_detections(
        detections=detections,
        parent_metadata=image.parent_metadata,
        parent_id_key=PARENT_ID_KEY,
        coordinates_key=PARENT_COORDINATES_KEY,
        dimensions_key=PARENT_DIMENSIONS_KEY,
    )


def attach_parent_coordinates_to_detections(
    detections: sv.Detections,
    parent_metadata: ImageParentMetadata,
    parent_id_key: str,
    coordinates_key: str,
    dimensions_key: str,
) -> sv.Detections:
    parent_coordinates_system = parent_metadata.origin_coordinates
    detections[parent_id_key] = np.array([parent_metadata.parent_id] * len(detections))
    coordinates = np.array(
        [[parent_coordinates_system.left_top_x, parent_coordinates_system.left_top_y]]
        * len(detections)
    )
    detections[coordinates_key] = coordinates
    dimensions = np.array(
        [
            [
                parent_coordinates_system.origin_height,
                parent_coordinates_system.origin_width,
            ]
        ]
        * len(detections)
    )
    detections[dimensions_key] = dimensions
    return detections


KEYS_REQUIRED_TO_EMBED_IN_ROOT_COORDINATES = {
    ROOT_PARENT_COORDINATES_KEY,
    ROOT_PARENT_DIMENSIONS_KEY,
    ROOT_PARENT_ID_KEY,
}


def sv_detections_to_root_coordinates(
    detections: sv.Detections, keypoints_key: str = KEYPOINTS_XY_KEY_IN_SV_DETECTIONS
) -> sv.Detections:
    detections_copy = deepcopy(detections)
    if len(detections_copy) == 0:
        return detections_copy

    if any(
        key not in detections_copy.data
        for key in KEYS_REQUIRED_TO_EMBED_IN_ROOT_COORDINATES
    ):
        logging.warning(
            "Could not execute detections_to_root_coordinates(...) on detections with "
            f"the following metadata registered: {list(detections_copy.data.keys())}"
        )
        return detections_copy
    if SCALING_RELATIVE_TO_ROOT_PARENT_KEY in detections_copy.data:
        scale = detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY][0]
        detections_copy = scale_sv_detections(
            detections=detections,
            scale=1 / scale,
        )
    detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = np.array(
        [1.0] * len(detections_copy)
    )
    detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = np.array(
        [1.0] * len(detections_copy)
    )
    origin_height = detections_copy[ROOT_PARENT_DIMENSIONS_KEY][0][0]
    origin_width = detections_copy[ROOT_PARENT_DIMENSIONS_KEY][0][1]
    detections_copy[IMAGE_DIMENSIONS_KEY] = np.array(
        [[origin_height, origin_width]] * len(detections_copy)
    )
    root_parent_id = detections_copy[ROOT_PARENT_ID_KEY][0]
    shift_x, shift_y = detections_copy[ROOT_PARENT_COORDINATES_KEY][0]
    detections_copy.xyxy += [shift_x, shift_y, shift_x, shift_y]
    if keypoints_key in detections_copy.data:
        for keypoints in detections_copy[keypoints_key]:
            if len(keypoints):
                keypoints += [shift_x, shift_y]
    if POLYGON_KEY_IN_SV_DETECTIONS in detections_copy.data:
        polygon_shift = np.asarray([shift_x, shift_y])
        detections_copy.data[POLYGON_KEY_IN_SV_DETECTIONS] = (
            detections_copy.data[POLYGON_KEY_IN_SV_DETECTIONS] + polygon_shift
        )
    if ORIENTED_BOX_COORDINATES in detections_copy.data:
        # crop localization subtracts the crop origin from the OBB corners
        # (dynamic_crop), so root conversion must add it back - same as xyxy,
        # keypoints and polygons above
        detections_copy.data[ORIENTED_BOX_COORDINATES] = detections_copy.data[
            ORIENTED_BOX_COORDINATES
        ] + np.asarray([shift_x, shift_y])
    if detections_copy.mask is not None:
        origin_mask_base = np.full((origin_height, origin_width), False)
        new_anchored_masks = np.array(
            [origin_mask_base.copy() for _ in detections_copy]
        )
        for anchored_mask, original_mask in zip(
            new_anchored_masks, detections_copy.mask
        ):
            mask_h, mask_w = original_mask.shape
            # TODO: instead of shifting mask we could store contours in data instead of storing mask (even if calculated)
            #       it would be faster to shift contours but at expense of having to remember to generate mask from contour when it's needed
            anchored_mask[shift_y : shift_y + mask_h, shift_x : shift_x + mask_w] = (
                original_mask
            )
        detections_copy.mask = new_anchored_masks
    new_root_metadata = ImageParentMetadata(
        parent_id=root_parent_id,
        origin_coordinates=OriginCoordinatesSystem(
            left_top_y=0,
            left_top_x=0,
            origin_width=origin_width,
            origin_height=origin_height,
        ),
    )
    detections_copy = attach_parent_coordinates_to_detections(
        detections=detections_copy,
        parent_metadata=new_root_metadata,
        parent_id_key=ROOT_PARENT_ID_KEY,
        coordinates_key=ROOT_PARENT_COORDINATES_KEY,
        dimensions_key=ROOT_PARENT_DIMENSIONS_KEY,
    )
    return attach_parent_coordinates_to_detections(
        detections=detections_copy,
        parent_metadata=new_root_metadata,
        parent_id_key=PARENT_ID_KEY,
        coordinates_key=PARENT_COORDINATES_KEY,
        dimensions_key=PARENT_DIMENSIONS_KEY,
    )


def filter_out_unwanted_classes_from_sv_detections_batch(
    predictions: List[sv.Detections],
    classes_to_accept: Optional[List[str]],
) -> List[sv.Detections]:
    if not classes_to_accept:
        return predictions
    filtered_predictions = []
    for prediction in predictions:
        filtered_prediction = prediction[
            np.isin(prediction[CLASS_NAME_DATA_FIELD], classes_to_accept)
        ]
        filtered_predictions.append(filtered_prediction)
    return filtered_predictions


def grab_batch_parameters(
    operations_parameters: Dict[str, Any],
    main_batch_size: int,
) -> Dict[str, Any]:
    return {
        key: value.broadcast(n=main_batch_size)
        for key, value in operations_parameters.items()
        if isinstance(value, Batch)
    }


def grab_non_batch_parameters(operations_parameters: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in operations_parameters.items()
        if not isinstance(value, Batch)
    }


def scale_sv_detections(
    detections: sv.Detections,
    scale: Union[float, Tuple[float, float]],
    keypoints_key: str = KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
    target_size_wh: Optional[Tuple[int, int]] = None,
    update_scaling_metadata: bool = True,
) -> sv.Detections:
    """Scale detection geometry into a new image coordinate frame.

    Args:
        detections: Predictions in the source image frame.
        scale: Isotropic factor, or ``(scale_x, scale_y)`` for aspect-preserving
            resizes where integer truncation makes the axes differ.
        keypoints_key: Key under which keypoint xy arrays are stored.
        target_size_wh: Optional exact ``(width, height)`` of the destination
            image (e.g. the JPEG that will be uploaded). When set, dense masks
            and ``image_dimensions`` are forced to this canvas so annotations
            stay intact relative to the stored image.
        update_scaling_metadata: Whether to update the scalar workflow coordinate
            lineage metadata. Set to ``False`` for terminal consumers, such as
            Dataset Upload, when using different X/Y scales. An anisotropic
            transform cannot be represented by the existing scalar metadata.
    """
    detections_copy = deepcopy(detections)
    if len(detections_copy) == 0:
        return detections_copy

    if isinstance(scale, (tuple, list)):
        scale_x, scale_y = float(scale[0]), float(scale[1])
    else:
        scale_x = scale_y = float(scale)

    scales_are_isotropic = abs(scale_x - scale_y) < 1e-9
    if update_scaling_metadata and not scales_are_isotropic:
        raise ValueError(
            "Anisotropic scaling cannot be represented by scalar workflow "
            "coordinate metadata. Set update_scaling_metadata=False for a "
            "terminal result that will not undergo root-coordinate recovery."
        )

    xyxy = detections_copy.xyxy.astype(np.float64, copy=True)
    xyxy[:, [0, 2]] *= scale_x
    xyxy[:, [1, 3]] *= scale_y
    detections_copy.xyxy = xyxy.round()

    if keypoints_key in detections_copy.data:
        for i in range(len(detections_copy[keypoints_key])):
            keypoints = detections_copy[keypoints_key][i]
            if len(keypoints) == 0:
                continue
            scaled_keypoints = keypoints.astype(np.float32, copy=True)
            scaled_keypoints[..., 0] *= scale_x
            scaled_keypoints[..., 1] *= scale_y
            detections_copy[keypoints_key][i] = scaled_keypoints.round()

    if target_size_wh is not None:
        target_w, target_h = int(target_size_wh[0]), int(target_size_wh[1])
        detections_copy[IMAGE_DIMENSIONS_KEY] = np.array(
            [[target_h, target_w]] * len(detections_copy)
        )
    elif IMAGE_DIMENSIONS_KEY in detections_copy.data:
        image_dimensions = detections_copy[IMAGE_DIMENSIONS_KEY].astype(
            np.float64, copy=True
        )
        image_dimensions[:, 0] *= scale_y
        image_dimensions[:, 1] *= scale_x
        detections_copy[IMAGE_DIMENSIONS_KEY] = image_dimensions.round()

    # RLE-only predictions (`mask=None` with the `rle_mask` data key) are
    # intentionally passed through untouched, matching historical behaviour: no
    # stock workflow routes them through a resize, and decoding full-resolution
    # RLE just to resize it is expensive. Their RLE stays sized to the source
    # canvas after scaling - callers needing scaled RLE must densify first.
    if detections_copy.mask is not None:
        # Resize dense masks directly onto the destination canvas. Polygon →
        # scale → raster round-trips fragment edge-touching masks; combined with
        # mask_to_polygon that used to take contours[0], Dataset Upload persisted
        # speckles instead of the real instance.
        original_mask_size_wh = (
            detections_copy.mask.shape[2],
            detections_copy.mask.shape[1],
        )
        if target_size_wh is not None:
            scaled_w, scaled_h = int(target_size_wh[0]), int(target_size_wh[1])
        else:
            scaled_w = int(round(original_mask_size_wh[0] * scale_x))
            scaled_h = int(round(original_mask_size_wh[1] * scale_y))
        scaled_w = max(1, scaled_w)
        scaled_h = max(1, scaled_h)
        if (scaled_w, scaled_h) != original_mask_size_wh:
            rle_masks_present = RLE_MASK_KEY_IN_SV_DETECTIONS in detections_copy.data
            if rle_masks_present:
                # pycocotools is not a base dependency - it ships with the extras
                # that produce RLE predictions - so keep the import deferred and
                # only reach it when RLE masks are actually involved.
                from pycocotools import mask as mask_utils

            resized_masks = np.array(
                [
                    cv2.resize(
                        detection_mask.astype(np.uint8),
                        (scaled_w, scaled_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
                    for detection_mask in detections_copy.mask
                ]
            )
            detections_copy.mask = resized_masks
            if rle_masks_present:
                # Source RLE counts encode the old canvas and cannot be scaled
                # arithmetically. Re-encode every resized mask, including valid
                # all-zero masks, to preserve detection-to-RLE alignment.
                resized_rle_masks = []
                for detection_mask in resized_masks:
                    rle_mask = mask_utils.encode(
                        np.asfortranarray(detection_mask.astype(np.uint8))
                    )
                    if isinstance(rle_mask["counts"], bytes):
                        rle_mask["counts"] = rle_mask["counts"].decode("utf-8")
                    resized_rle_masks.append(rle_mask)
                detections_copy.data[RLE_MASK_KEY_IN_SV_DETECTIONS] = np.array(
                    resized_rle_masks, dtype=object
                )

    if POLYGON_KEY_IN_SV_DETECTIONS in detections_copy.data:
        polygons = detections_copy.data[POLYGON_KEY_IN_SV_DETECTIONS]
        if isinstance(polygons, np.ndarray) and np.issubdtype(
            polygons.dtype, np.number
        ):
            scaled_polygons = polygons.astype(np.float64, copy=True)
            scaled_polygons[..., 0] *= scale_x
            scaled_polygons[..., 1] *= scale_y
            detections_copy.data[POLYGON_KEY_IN_SV_DETECTIONS] = (
                scaled_polygons.round().astype(np.int32)
            )
        else:
            # Ragged object-dtype polygon lists
            scaled_polygons = []
            for polygon in polygons:
                scaled_polygon = np.asarray(polygon, dtype=np.float64).copy()
                scaled_polygon[..., 0] *= scale_x
                scaled_polygon[..., 1] *= scale_y
                scaled_polygons.append(scaled_polygon.round().astype(np.int32))
            detections_copy.data[POLYGON_KEY_IN_SV_DETECTIONS] = np.array(
                scaled_polygons, dtype=object
            )

    if update_scaling_metadata:
        if SCALING_RELATIVE_TO_PARENT_KEY in detections_copy.data:
            detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = (
                detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] * scale_x
            )
        else:
            detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = np.array(
                [scale_x] * len(detections_copy)
            )
        if SCALING_RELATIVE_TO_ROOT_PARENT_KEY in detections_copy.data:
            detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = (
                detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] * scale_x
            )
        else:
            detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = np.array(
                [scale_x] * len(detections_copy)
            )
    return detections_copy


def remove_unexpected_keys_from_dictionary(
    dictionary: dict,
    expected_keys: set,
) -> dict:
    """This function mutates input `dictionary`"""
    unexpected_keys = set(dictionary.keys()).difference(expected_keys)
    for unexpected_key in unexpected_keys:
        del dictionary[unexpected_key]
    return dictionary


def post_process_ocr_result(
    images: Batch[WorkflowImageData],
    predictions: List[dict],
    expected_output_keys: Set[str],
) -> BlockResult:
    for prediction, image in zip(predictions, images):
        raw_predictions = prediction.get("predictions", [])
        prediction["predictions"] = sv.Detections.from_inference(prediction)
        if len(prediction["predictions"]) != len(raw_predictions):
            raw_predictions = filter_out_invalid_polygons(predictions=raw_predictions)
        detection_ids = [_get_or_create_detection_id(p) for p in raw_predictions]
        prediction["predictions"]["detection_id"] = detection_ids
        prediction[PREDICTION_TYPE_KEY] = "ocr"
        prediction[PARENT_ID_KEY] = image.parent_metadata.parent_id
        prediction[ROOT_PARENT_ID_KEY] = image.workflow_root_ancestor_metadata.parent_id
        _ = remove_unexpected_keys_from_dictionary(
            dictionary=prediction,
            expected_keys=expected_output_keys,
        )
    return predictions


def run_in_parallel(tasks: List[Callable[[], T]], max_workers: int = 1) -> List[T]:
    tasks = [wrap_with_context_snapshot(task) for task in tasks]
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda f: f(), tasks))


DETECTION_MAX_EDGE_PIXELS = 2048
"""Maximum longest edge used when uploading images for VLM object detection."""


def scale_dimensions_to_max_edge(
    width: int,
    height: int,
    max_edge: int,
) -> Tuple[int, int]:
    """Scale dimensions down so the longest edge is at most ``max_edge``.

    Never upscales and preserves the aspect ratio. Both the VLM blocks that
    downscale images before upload and the parsers that map returned pixel
    coordinates back onto the original image must use this exact arithmetic.

    Args:
        width: Original image width in pixels.
        height: Original image height in pixels.
        max_edge: Maximum allowed longest edge in pixels.

    Returns:
        Target ``(width, height)`` after scaling.
    """
    if max(width, height) <= max_edge:
        return (width, height)

    if width >= height:
        scaled_width = max_edge
        scaled_height = max(round(height * max_edge / width), 1)
    else:
        scaled_height = max_edge
        scaled_width = max(round(width * max_edge / height), 1)

    return (scaled_width, scaled_height)


ANTHROPIC_DETECTION_MAX_EDGE_PIXELS = 2576
"""Maximum padded edge length of images uploaded to Anthropic Claude models.

High-resolution tier limit from Anthropic's vision documentation; edges are
padded up to a multiple of ``ANTHROPIC_IMAGE_TILE_PIXELS`` before the check.
"""

ANTHROPIC_DETECTION_MAX_IMAGE_TOKENS = 4784
"""Maximum visual-token budget of images uploaded to Anthropic Claude models."""

ANTHROPIC_IMAGE_TILE_PIXELS = 28
"""Edge length of the square tiles Claude tokenizes images with."""


def count_anthropic_image_tokens(width: int, height: int) -> int:
    """Count the visual tokens Claude spends on an image of given dimensions.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Number of visual tokens.
    """
    tile = ANTHROPIC_IMAGE_TILE_PIXELS
    return math.ceil(width / tile) * math.ceil(height / tile)


def compute_anthropic_upload_dimensions(
    width: int,
    height: int,
    max_edge: int = ANTHROPIC_DETECTION_MAX_EDGE_PIXELS,
    max_tokens: int = ANTHROPIC_DETECTION_MAX_IMAGE_TOKENS,
) -> Tuple[int, int]:
    """Compute the dimensions Claude resizes an image to before processing.

    Mirrors the reference implementation from Anthropic's vision coordinates
    documentation, so pixel coordinates returned by Claude map one-to-one onto
    an image pre-resized to these dimensions. Never upscales. The Claude block
    that pre-resizes detection uploads and the parser that maps returned pixel
    coordinates back onto the original image must use this exact arithmetic.

    Args:
        width: Original image width in pixels.
        height: Original image height in pixels.
        max_edge: Maximum padded edge length in pixels.
        max_tokens: Maximum visual-token budget.

    Returns:
        Target ``(width, height)`` after Claude's internal resize.
    """
    tile = ANTHROPIC_IMAGE_TILE_PIXELS

    def fits(candidate_width: int, candidate_height: int) -> bool:
        return (
            math.ceil(candidate_width / tile) * tile <= max_edge
            and math.ceil(candidate_height / tile) * tile <= max_edge
            and count_anthropic_image_tokens(candidate_width, candidate_height)
            <= max_tokens
        )

    if fits(width, height):
        return (width, height)

    if height > width:
        resized_height, resized_width = compute_anthropic_upload_dimensions(
            height, width, max_edge, max_tokens
        )
        return (resized_width, resized_height)

    aspect_ratio = width / height
    low, high = 1, width
    while low + 1 < high:
        mid = (low + high) // 2
        if fits(mid, max(round(mid / aspect_ratio), 1)):
            low = mid
        else:
            high = mid
    return (low, max(round(low / aspect_ratio), 1))
