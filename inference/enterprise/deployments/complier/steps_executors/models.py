from functools import partial
from typing import Any, Dict, Optional, Tuple, Union

from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    InferenceRequest,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.managers.base import ModelManager
from inference.enterprise.deployments.complier.steps_executors.types import (
    NextStepReference,
    OutputsLookup,
)
from inference.enterprise.deployments.complier.utils import (
    construct_step_selector,
    get_last_selector_chunk,
    get_step_selector_from_its_output,
    is_input_selector,
    is_step_output_selector,
)
from inference.enterprise.deployments.entities.steps import (
    ClassificationModel,
    Crop,
    InstanceSegmentationModel,
    KeypointsDetectionModel,
    MultiLabelClassificationModel,
    ObjectDetectionModel,
    OCRModel,
    RoboflowModel,
    is_selector,
)


async def run_roboflow_model_step(
    step: RoboflowModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    model_manager.add_model(
        model_id=step.model_id,
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
    result = await model_manager.infer_from_request(
        model_id=step.model_id, request=request
    )
    if issubclass(type(result), list):
        serialised_result = [e.dict() for e in result]
    else:
        serialised_result = result.dict()
    if issubclass(type(serialised_result), list) and len(serialised_result) == 1:
        serialised_result = serialised_result[0]
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


def resolve_parameter(
    selector_or_value: Any,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> Any:
    if not is_selector(selector_or_value=selector_or_value):
        return selector_or_value
    if is_step_output_selector(selector_or_value=selector_or_value):
        step_selector = get_step_selector_from_its_output(
            step_output_selector=selector_or_value
        )
        step_output = outputs_lookup[step_selector]
        return step_output[get_last_selector_chunk(selector=selector_or_value)]
    return runtime_parameters[get_last_selector_chunk(selector=selector_or_value)]


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
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


def load_core_model(
    model_manager: ModelManager,
    inference_request: DoctrOCRInferenceRequest,
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


def get_image(
    step: Union[RoboflowModel, OCRModel],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> Any:
    if is_input_selector(selector_or_value=step.image):
        image = runtime_parameters[get_last_selector_chunk(selector=step.image)]
    elif is_step_output_selector(selector_or_value=step.image):
        step_selector = get_step_selector_from_its_output(
            step_output_selector=step.image
        )
        step_output = outputs_lookup[step_selector]
        image = step_output[get_last_selector_chunk(selector=step.image)]
    else:
        raise RuntimeError("Cannot find image")
    return image


async def run_crop_step(
    step: Crop,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> Tuple[NextStepReference, OutputsLookup]:
    pass
