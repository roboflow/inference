import logging
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
from supervision.config import CLASS_NAME_DATA_FIELD
from supervision.detection.compact_mask import CompactMask

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.easy_ocr import EasyOCRInferenceRequest
from inference.core.entities.requests.sam2 import Sam2InferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.managers.base import ModelManager
from inference.core.roboflow_api import ModelEndpointType
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


def _mask_crop_to_compact_rle_counts(mask: np.ndarray) -> np.ndarray:
    flat = np.asarray(mask, dtype=np.bool_).ravel(order="F")
    if len(flat) == 0:
        return np.array([0], dtype=np.int32)
    changes = np.diff(flat.view(np.uint8))
    boundaries = np.where(changes != 0)[0] + 1
    positions = np.concatenate(([0], boundaries, [len(flat)]))
    run_lengths = np.diff(positions).astype(np.int32)
    if flat[0]:
        run_lengths = np.concatenate(([np.int32(0)], run_lengths))
    return run_lengths


def _polygon_prediction_to_compact_mask_crop(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    x_min = int(np.min(polygon[:, 0]))
    x_max = int(np.max(polygon[:, 0]))
    y_min = int(np.min(polygon[:, 1]))
    y_max = int(np.max(polygon[:, 1]))

    if x_max < 0 or y_max < 0 or x_min >= image_width or y_min >= image_height:
        mask = np.zeros((1, 1), dtype=bool)
        return (
            _mask_crop_to_compact_rle_counts(mask),
            (1, 1),
            (
                min(max(x_min, 0), image_width - 1),
                min(max(y_min, 0), image_height - 1),
            ),
        )

    x1 = max(0, x_min)
    y1 = max(0, y_min)
    x2 = min(image_width - 1, x_max)
    y2 = min(image_height - 1, y_max)
    crop = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    shifted_polygon = polygon - np.array([x1, y1], dtype=np.int32)
    cv2.fillPoly(crop, [shifted_polygon], color=(1,))
    return (
        _mask_crop_to_compact_rle_counts(crop),
        (crop.shape[0], crop.shape[1]),
        (x1, y1),
    )


def _try_convert_polygon_predictions_to_sv_detections(
    prediction: Dict[str, Union[List[Dict[str, Any]], Any]],
    predictions_key: str,
    image_key: str,
) -> Optional[Tuple[sv.Detections, List[Dict[str, Any]]]]:
    raw_predictions = prediction[predictions_key]
    required_prediction_keys = {
        X_KEY,
        Y_KEY,
        WIDTH_KEY,
        HEIGHT_KEY,
        "confidence",
        "class_id",
        "class",
        "points",
    }
    if any(
        not required_prediction_keys.issubset(p)
        or p.get("rle") is not None
        or p.get(RLE_MASK_KEY_IN_INFERENCE_RESPONSE) is not None
        for p in raw_predictions
    ):
        return None

    has_tracker = ["tracker_id" in p for p in raw_predictions]
    if any(has_tracker) and not all(has_tracker):
        return None

    image_width = int(prediction[image_key][WIDTH_KEY])
    image_height = int(prediction[image_key][HEIGHT_KEY])
    valid_predictions = filter_out_invalid_polygons(predictions=raw_predictions)
    if not valid_predictions:
        detections = sv.Detections.empty()
        detections.data = {CLASS_NAME_DATA_FIELD: np.empty(0, dtype=str)}
        return detections, valid_predictions

    count = len(valid_predictions)
    xyxy = np.empty((count, 4), dtype=np.float64)
    confidence = np.empty(count, dtype=np.float64)
    class_id = np.empty(count, dtype=np.int64)
    class_name = []
    tracker_id = np.empty(count, dtype=np.int64) if all(has_tracker) else None
    rles = []
    crop_shapes = np.empty((count, 2), dtype=np.int32)
    offsets = np.empty((count, 2), dtype=np.int32)

    for idx, item in enumerate(valid_predictions):
        x = float(item[X_KEY])
        y = float(item[Y_KEY])
        width = float(item[WIDTH_KEY])
        height = float(item[HEIGHT_KEY])
        x_min = x - width / 2
        y_min = y - height / 2
        xyxy[idx] = [x_min, y_min, x_min + width, y_min + height]
        confidence[idx] = float(item["confidence"])
        class_id[idx] = int(item["class_id"])
        class_name.append(item["class"])
        if tracker_id is not None:
            tracker_id[idx] = int(item["tracker_id"])

        polygon = np.array(
            [[point[X_KEY], point[Y_KEY]] for point in item["points"]],
            dtype=np.int32,
        )
        rle, crop_shape, offset = _polygon_prediction_to_compact_mask_crop(
            polygon=polygon,
            image_width=image_width,
            image_height=image_height,
        )
        rles.append(rle)
        crop_shapes[idx] = crop_shape
        offsets[idx] = offset

    masks = CompactMask(
        rles=rles,
        crop_shapes=crop_shapes,
        offsets=offsets,
        image_shape=(image_height, image_width),
    )
    detections = sv.Detections(
        xyxy=xyxy,
        confidence=confidence,
        class_id=class_id,
        mask=masks,
        tracker_id=tracker_id,
        data={CLASS_NAME_DATA_FIELD: np.array(class_name)},
    )
    return detections, valid_predictions


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
        fast_result = _try_convert_polygon_predictions_to_sv_detections(
            prediction=p,
            predictions_key=predictions_key,
            image_key=image_key,
        )
        if fast_result is None:
            detections = sv.Detections.from_inference(p)
            raw_predictions = p[predictions_key]
            if len(detections) != len(raw_predictions):
                raw_predictions = filter_out_invalid_polygons(
                    predictions=raw_predictions
                )
        else:
            detections, raw_predictions = fast_result

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
    padded_class_name = np.full((n, max_kps), "", dtype=object)
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
    scale: float,
    keypoints_key: str = KEYPOINTS_XY_KEY_IN_SV_DETECTIONS,
) -> sv.Detections:
    detections_copy = deepcopy(detections)
    if len(detections_copy) == 0:
        return detections_copy
    detections_copy.xyxy = (detections_copy.xyxy * scale).round()
    if keypoints_key in detections_copy.data:
        for i in range(len(detections_copy[keypoints_key])):
            detections_copy[keypoints_key][i] = (
                detections_copy[keypoints_key][i].astype(np.float32) * scale
            ).round()
    detections_copy[IMAGE_DIMENSIONS_KEY] = (
        detections_copy[IMAGE_DIMENSIONS_KEY] * scale
    ).round()
    if detections_copy.mask is not None:
        scaled_masks = []
        original_mask_size_wh = (
            detections_copy.mask.shape[2],
            detections_copy.mask.shape[1],
        )
        scaled_mask_size_wh = round(original_mask_size_wh[0] * scale), round(
            original_mask_size_wh[1] * scale
        )
        for detection_mask in detections_copy.mask:
            polygons = sv.mask_to_polygons(mask=detection_mask)
            polygon_masks = []
            for polygon in polygons:
                scaled_polygon = (polygon * scale).round().astype(np.int32)
                polygon_masks.append(
                    sv.polygon_to_mask(
                        polygon=scaled_polygon, resolution_wh=scaled_mask_size_wh
                    )
                )
            scaled_detection_mask = np.sum(polygon_masks, axis=0) > 0
            scaled_masks.append(scaled_detection_mask)
        detections_copy.mask = np.array(scaled_masks)
    if POLYGON_KEY_IN_SV_DETECTIONS in detections_copy.data:
        detections_copy.data[POLYGON_KEY_IN_SV_DETECTIONS] = (
            (detections_copy.data[POLYGON_KEY_IN_SV_DETECTIONS] * scale)
            .round()
            .astype(np.int32)
        )
    if SCALING_RELATIVE_TO_PARENT_KEY in detections_copy.data:
        detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = (
            detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] * scale
        )
    else:
        detections_copy[SCALING_RELATIVE_TO_PARENT_KEY] = np.array(
            [scale] * len(detections_copy)
        )
    if SCALING_RELATIVE_TO_ROOT_PARENT_KEY in detections_copy.data:
        detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = (
            detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] * scale
        )
    else:
        detections_copy[SCALING_RELATIVE_TO_ROOT_PARENT_KEY] = np.array(
            [scale] * len(detections_copy)
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
    tasks = _propagate_inference_context(tasks)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(lambda f: f(), tasks))


def _propagate_inference_context(
    tasks: List[Callable[[], T]],
) -> List[Callable[[], T]]:
    """Wrap each task so that inference_sdk context vars and server-side
    context vars are propagated into worker threads.  Returns the tasks
    unchanged when no context is active.
    """
    from inference.core.managers.model_load_collector import (
        model_load_info,
        request_model_ids,
    )

    load_collector = model_load_info.get(None)
    ids_collector = request_model_ids.get(None)

    try:
        from asgi_correlation_id import correlation_id as _cid_ctx

        corr_id = _cid_ctx.get()
    except Exception:
        corr_id = None

    try:
        from inference_sdk.config import (
            apply_duration_minimum,
            execution_id,
            remote_processing_times,
        )

        exec_id = execution_id.get() if execution_id is not None else None
        rpt_collector = (
            remote_processing_times.get()
            if remote_processing_times is not None
            else None
        )
        duration_min = (
            apply_duration_minimum.get() if apply_duration_minimum is not None else None
        )
    except ImportError:
        exec_id = None
        rpt_collector = None
        duration_min = None

    if (
        exec_id is None
        and rpt_collector is None
        and duration_min is None
        and load_collector is None
        and ids_collector is None
        and corr_id is None
    ):
        return tasks

    def _wrap(fun: Callable[[], T]) -> Callable[[], T]:
        def _with_context() -> T:
            if corr_id is not None:
                from asgi_correlation_id import correlation_id as _cid_ctx

                _cid_ctx.set(corr_id)
            if exec_id is not None:
                from inference_sdk.config import execution_id

                execution_id.set(exec_id)
            if rpt_collector is not None:
                from inference_sdk.config import remote_processing_times

                remote_processing_times.set(rpt_collector)
            if duration_min is not None:
                from inference_sdk.config import apply_duration_minimum

                apply_duration_minimum.set(duration_min)
            if load_collector is not None:
                model_load_info.set(load_collector)
            if ids_collector is not None:
                request_model_ids.set(ids_collector)
            return fun()

        return _with_context

    return [_wrap(t) for t in tasks]
