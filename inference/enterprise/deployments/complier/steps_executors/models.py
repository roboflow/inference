from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.managers.base import ModelManager
from inference.enterprise.deployments.complier.steps_executors.constants import (
    CENTER_X_KEY,
    CENTER_Y_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_COORDINATES_SUFFIX,
)
from inference.enterprise.deployments.complier.steps_executors.types import (
    NextStepReference,
    OutputsLookup,
)
from inference.enterprise.deployments.complier.steps_executors.utils import (
    get_image,
    resolve_parameter,
)
from inference.enterprise.deployments.complier.utils import construct_step_selector
from inference.enterprise.deployments.entities.steps import (
    ClassificationModel,
    ClipComparison,
    InstanceSegmentationModel,
    KeypointsDetectionModel,
    MultiLabelClassificationModel,
    ObjectDetectionModel,
    OCRModel,
    RoboflowModel,
)


async def run_roboflow_model_step(
    step: RoboflowModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    model_id = resolve_parameter(
        selector_or_value=step.model_id,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    model_manager.add_model(
        model_id=model_id,
        api_key=api_key,
    )
    image = get_image(
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    request_constructor = MODEL_TYPE2REQUEST_CONSTRUCTOR[step.type]
    request = request_constructor(
        step=step,
        image=image,
        api_key=api_key,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    result = await model_manager.infer_from_request(model_id=model_id, request=request)
    if issubclass(type(result), list):
        serialised_result = [e.dict(by_alias=True, exclude_none=True) for e in result]
    else:
        serialised_result = result.dict(by_alias=True, exclude_none=True)
    if issubclass(type(serialised_result), list) and len(serialised_result) == 1:
        serialised_result = serialised_result[0]
    if issubclass(type(image), list) and len(image) == 1:
        image = image[0]
    if step.type in {"ClassificationModel", "MultiLabelClassificationModel"}:
        serialised_result = attach_parent_info(
            image=image, results=serialised_result, nested_key=None
        )
    else:
        serialised_result = attach_parent_info(image=image, results=serialised_result)
        serialised_result = anchor_detections_in_parent_coordinates(
            image=image,
            serialised_result=serialised_result,
        )
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


def construct_classification_request(
    step: Union[ClassificationModel, MultiLabelClassificationModel],
    image: Any,
    api_key: Optional[str],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> ClassificationInferenceRequest:
    resolve = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    return ClassificationInferenceRequest(
        api_key=api_key,
        model_id=resolve(step.model_id),
        image=image,
        confidence=resolve(step.confidence),
        disable_active_learning=resolve(step.disable_active_learning),
    )


def construct_object_detection_request(
    step: ObjectDetectionModel,
    image: Any,
    api_key: Optional[str],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> ObjectDetectionInferenceRequest:
    resolve = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    return ObjectDetectionInferenceRequest(
        api_key=api_key,
        model_id=resolve(step.model_id),
        image=image,
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
    )


def construct_instance_segmentation_request(
    step: InstanceSegmentationModel,
    image: Any,
    api_key: Optional[str],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> InstanceSegmentationInferenceRequest:
    resolve = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    return InstanceSegmentationInferenceRequest(
        api_key=api_key,
        model_id=resolve(step.model_id),
        image=image,
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        mask_decode_mode=resolve(step.mask_decode_mode),
        tradeoff_factor=resolve(step.tradeoff_factor),
    )


def construct_keypoints_detection_request(
    step: KeypointsDetectionModel,
    image: Any,
    api_key: Optional[str],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> KeypointsDetectionInferenceRequest:
    resolve = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    return KeypointsDetectionInferenceRequest(
        api_key=api_key,
        model_id=resolve(step.model_id),
        image=image,
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        keypoint_confidence=resolve(step.keypoint_confidence),
    )


MODEL_TYPE2REQUEST_CONSTRUCTOR = {
    "ClassificationModel": construct_classification_request,
    "MultiLabelClassificationModel": construct_classification_request,
    "ObjectDetectionModel": construct_object_detection_request,
    "KeypointsDetectionModel": construct_keypoints_detection_request,
    "InstanceSegmentationModel": construct_instance_segmentation_request,
}


async def run_ocr_model_step(
    step: OCRModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    image = get_image(
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    if not issubclass(type(image), list):
        image = [image]
    serialised_result = []
    for single_image in image:
        inference_request = DoctrOCRInferenceRequest(
            image=single_image,
        )
        doctr_model_id = load_core_model(
            model_manager=model_manager,
            inference_request=inference_request,
            core_model="doctr",
            api_key=api_key,
        )
        result = await model_manager.infer_from_request(
            doctr_model_id, inference_request
        )
        serialised_result.append(result.dict())
    if len(serialised_result) == 1:
        serialised_result = serialised_result[0]
        image = image[0]
    serialised_result = attach_parent_info(
        image=image,
        results=serialised_result,
        nested_key=None,
    )
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


async def run_clip_comparison_step(
    step: ClipComparison,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    image = get_image(
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    text = resolve_parameter(
        selector_or_value=step.text,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    if not issubclass(type(image), list):
        image = [image]
    serialised_result = []
    for single_image in image:
        inference_request = ClipCompareRequest(
            subject=single_image, subject_type="image", prompt=text, prompt_type="text"
        )
        doctr_model_id = load_core_model(
            model_manager=model_manager,
            inference_request=inference_request,
            core_model="clip",
            api_key=api_key,
        )
        result = await model_manager.infer_from_request(
            doctr_model_id, inference_request
        )
        serialised_result.append(result.dict())
    if len(serialised_result) == 1:
        serialised_result = serialised_result[0]
        image = image[0]
    serialised_result = attach_parent_info(
        image=image,
        results=serialised_result,
        nested_key=None,
    )
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


def load_core_model(
    model_manager: ModelManager,
    inference_request: Union[DoctrOCRInferenceRequest, ClipCompareRequest],
    core_model: str,
    api_key: Optional[str] = None,
) -> str:
    if api_key:
        inference_request.api_key = api_key
    version_id_field = f"{core_model}_version_id"
    core_model_id = (
        f"{core_model}/{inference_request.__getattribute__(version_id_field)}"
    )
    model_manager.add_model(core_model_id, inference_request.api_key)
    return core_model_id


def attach_parent_info(
    image: Union[Dict[str, Any], List[Dict[str, Any]]],
    results: Union[Dict[str, Any], List[Dict[str, Any]]],
    nested_key: Optional[str] = "predictions",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if issubclass(type(image), list):
        return [
            attach_parent_info_to_image_detections(
                image=i, predictions=p, nested_key=nested_key
            )
            for i, p in zip(image, results)
        ]
    return attach_parent_info_to_image_detections(
        image=image, predictions=results, nested_key=nested_key
    )


def attach_parent_info_to_image_detections(
    image: Dict[str, Any],
    predictions: Dict[str, Any],
    nested_key: Optional[str],
) -> Dict[str, Any]:
    predictions["parent_id"] = image["parent_id"]
    if nested_key is None:
        return predictions
    for prediction in predictions[nested_key]:
        prediction["parent_id"] = image["parent_id"]
    return predictions


def anchor_detections_in_parent_coordinates(
    image: Union[Dict[str, Any], List[Dict[str, Any]]],
    serialised_result: Union[Dict[str, Any], List[Dict[str, Any]]],
    image_metadata_key: str = "image",
    detections_key: str = "predictions",
) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    if issubclass(type(image), dict):
        return anchor_image_detections_in_parent_coordinates(
            image=image,
            serialised_result=serialised_result,
            image_metadata_key=image_metadata_key,
            detections_key=detections_key,
        )
    return [
        anchor_image_detections_in_parent_coordinates(
            image=i,
            serialised_result=d,
            image_metadata_key=image_metadata_key,
            detections_key=detections_key,
        )
        for i, d in zip(image, serialised_result)
    ]


def anchor_image_detections_in_parent_coordinates(
    image: Dict[str, Any],
    serialised_result: Dict[str, Any],
    image_metadata_key: str = "image",
    detections_key: str = "predictions",
) -> Dict[str, Any]:
    serialised_result[f"{detections_key}{PARENT_COORDINATES_SUFFIX}"] = deepcopy(
        serialised_result[detections_key]
    )
    serialised_result[f"{image_metadata_key}{PARENT_COORDINATES_SUFFIX}"] = deepcopy(
        serialised_result[image_metadata_key]
    )
    if ORIGIN_COORDINATES_KEY not in image:
        return serialised_result
    shift_x, shift_y = (
        image[ORIGIN_COORDINATES_KEY][CENTER_X_KEY],
        image[ORIGIN_COORDINATES_KEY][CENTER_Y_KEY],
    )
    for detection in serialised_result[f"{detections_key}{PARENT_COORDINATES_SUFFIX}"]:
        detection["x"] += shift_x
        detection["y"] += shift_y
    serialised_result[f"{image_metadata_key}{PARENT_COORDINATES_SUFFIX}"] = image[
        ORIGIN_COORDINATES_KEY
    ][ORIGIN_SIZE_KEY]
    return serialised_result
