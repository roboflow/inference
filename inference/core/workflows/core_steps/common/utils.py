import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, TypeVar, Union

import numpy as np
import supervision as sv
from supervision.config import CLASS_NAME_DATA_FIELD

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.gaze import GazeDetectionInferenceRequest
from inference.core.entities.requests.sam2 import Sam2InferenceRequest
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.managers.base import ModelManager
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
    PREDICTION_TYPE_KEY,
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

T = TypeVar("T")


def load_core_model(
    model_manager: ModelManager,
    inference_request: Union[
        DoctrOCRInferenceRequest,
        ClipCompareRequest,
        YOLOWorldInferenceRequest,
        Sam2InferenceRequest,
        GazeDetectionInferenceRequest,
    ],
    core_model: str,
) -> str:
    version_id_field = f"{core_model}_version_id"
    core_model_id = (
        f"{core_model}/{inference_request.__getattribute__(version_id_field)}"
    )
    model_manager.add_model(core_model_id, inference_request.api_key)
    return core_model_id


def attach_prediction_type_info(
    predictions: List[Dict[str, Any]],
    prediction_type: str,
    key: str = PREDICTION_TYPE_KEY,
) -> List[Dict[str, Any]]:
    for result in predictions:
        result[key] = prediction_type
    return predictions


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
        parent_ids = [d.get(PARENT_ID_KEY, "") for d in p[predictions_key]]
        detection_ids = [
            d.get(DETECTION_ID_KEY, str(uuid.uuid4())) for d in p[predictions_key]
        ]
        detections[DETECTION_ID_KEY] = np.array(detection_ids)
        detections[PARENT_ID_KEY] = np.array(parent_ids)
        detections[IMAGE_DIMENSIONS_KEY] = np.array([[height, width]] * len(detections))
        if INFERENCE_ID_KEY in p:
            detections[INFERENCE_ID_KEY] = np.array(
                [p[INFERENCE_ID_KEY]] * len(detections)
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
            np.array(
                [k[KEYPOINTS_CLASS_NAME_KEY_IN_INFERENCE_RESPONSE] for k in keypoints]
            )
        )
        keypoints_class_ids.append(
            np.array(
                [k[KEYPOINTS_CLASS_ID_KEY_IN_INFERENCE_RESPONSE] for k in keypoints]
            )
        )
        keypoints_confidences.append(
            np.array(
                [k[KEYPOINTS_CONFIDENCE_KEY_IN_INFERENCE_RESPONSE] for k in keypoints],
                dtype=np.float32,
            )
        )
        keypoints_xy.append(
            np.array([[k[X_KEY], k[Y_KEY]] for k in keypoints], dtype=np.float32)
        )
    detections[KEYPOINTS_CLASS_NAME_KEY_IN_SV_DETECTIONS] = np.array(
        keypoints_class_names, dtype="object"
    )
    detections[KEYPOINTS_CLASS_ID_KEY_IN_SV_DETECTIONS] = np.array(
        keypoints_class_ids, dtype="object"
    )
    detections[KEYPOINTS_CONFIDENCE_KEY_IN_SV_DETECTIONS] = np.array(
        keypoints_confidences, dtype="object"
    )
    detections[KEYPOINTS_XY_KEY_IN_SV_DETECTIONS] = np.array(
        keypoints_xy, dtype="object"
    )
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


def run_in_parallel(tasks: List[Callable[[], T]], max_workers: int = 1) -> List[T]:
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_run, tasks))


def _run(fun: Callable[[], T]) -> T:
    return fun()
