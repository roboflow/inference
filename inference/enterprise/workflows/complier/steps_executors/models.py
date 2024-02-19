import asyncio
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.doctr import DoctrOCRInferenceRequest
from inference.core.entities.requests.inference import (
    ClassificationInferenceRequest,
    InstanceSegmentationInferenceRequest,
    KeypointsDetectionInferenceRequest,
    ObjectDetectionInferenceRequest,
)
from inference.core.entities.requests.yolo_world import YOLOWorldInferenceRequest
from inference.core.env import (
    HOSTED_CLASSIFICATION_URL,
    HOSTED_CORE_MODEL_URL,
    HOSTED_DETECT_URL,
    HOSTED_INSTANCE_SEGMENTATION_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_TARGET,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
    WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
)
from inference.core.managers.base import ModelManager
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.steps_executors.constants import (
    CENTER_X_KEY,
    CENTER_Y_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_COORDINATES_SUFFIX,
)
from inference.enterprise.workflows.complier.steps_executors.types import (
    NextStepReference,
    OutputsLookup,
)
from inference.enterprise.workflows.complier.steps_executors.utils import (
    get_image,
    make_batches,
    resolve_parameter,
)
from inference.enterprise.workflows.complier.utils import construct_step_selector
from inference.enterprise.workflows.entities.steps import (
    ClassificationModel,
    ClipComparison,
    InstanceSegmentationModel,
    KeypointsDetectionModel,
    MultiLabelClassificationModel,
    ObjectDetectionModel,
    OCRModel,
    RoboflowModel,
    StepInterface,
    YoloWorld,
)
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

MODEL_TYPE2PREDICTION_TYPE = {
    "ClassificationModel": "classification",
    "MultiLabelClassificationModel": "classification",
    "ObjectDetectionModel": "object-detection",
    "InstanceSegmentationModel": "instance-segmentation",
    "KeypointsDetectionModel": "keypoint-detection",
}


async def run_roboflow_model_step(
    step: RoboflowModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
) -> Tuple[NextStepReference, OutputsLookup]:
    model_id = resolve_parameter(
        selector_or_value=step.model_id,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    image = get_image(
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    if step_execution_mode is StepExecutionMode.LOCAL:
        serialised_result = await get_roboflow_model_predictions_locally(
            image=image,
            model_id=model_id,
            step=step,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
            model_manager=model_manager,
            api_key=api_key,
        )
    else:
        serialised_result = await get_roboflow_model_predictions_from_remote_api(
            image=image,
            model_id=model_id,
            step=step,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
            api_key=api_key,
        )
    serialised_result = attach_prediction_type_info(
        results=serialised_result,
        prediction_type=MODEL_TYPE2PREDICTION_TYPE[step.get_type()],
    )
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


async def get_roboflow_model_predictions_locally(
    image: List[dict],
    model_id: str,
    step: RoboflowModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> List[dict]:
    request_constructor = MODEL_TYPE2REQUEST_CONSTRUCTOR[step.type]
    request = request_constructor(
        step=step,
        image=image,
        api_key=api_key,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    model_manager.add_model(
        model_id=model_id,
        api_key=api_key,
    )
    result = await model_manager.infer_from_request(model_id=model_id, request=request)
    if issubclass(type(result), list):
        serialised_result = [e.dict(by_alias=True, exclude_none=True) for e in result]
    else:
        serialised_result = [result.dict(by_alias=True, exclude_none=True)]
    return serialised_result


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
        disable_active_learning=resolve(step.disable_active_learning),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
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
        disable_active_learning=resolve(step.disable_active_learning),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
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
        disable_active_learning=resolve(step.disable_active_learning),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
        keypoint_confidence=resolve(step.keypoint_confidence),
    )


MODEL_TYPE2REQUEST_CONSTRUCTOR = {
    "ClassificationModel": construct_classification_request,
    "MultiLabelClassificationModel": construct_classification_request,
    "ObjectDetectionModel": construct_object_detection_request,
    "InstanceSegmentationModel": construct_instance_segmentation_request,
    "KeypointsDetectionModel": construct_keypoints_detection_request,
}


async def get_roboflow_model_predictions_from_remote_api(
    image: List[dict],
    model_id: str,
    step: RoboflowModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    api_key: Optional[str],
) -> List[dict]:
    api_url = resolve_model_api_url(step=step)
    client = InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key,
    )
    if WORKFLOWS_REMOTE_API_TARGET == "hosted":
        client.select_api_v0()
    configuration = MODEL_TYPE2HTTP_CLIENT_CONSTRUCTOR[step.type](
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    client.configure(inference_configuration=configuration)
    inference_input = [i["value"] for i in image]
    results = await client.infer_async(
        inference_input=inference_input,
        model_id=model_id,
    )
    if not issubclass(type(results), list):
        return [results]
    return results


def construct_http_client_configuration_for_classification_step(
    step: Union[ClassificationModel, MultiLabelClassificationModel],
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> InferenceConfiguration:
    resolve = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    return InferenceConfiguration(
        confidence_threshold=resolve(step.confidence),
        disable_active_learning=resolve(step.disable_active_learning),
        max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
        max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
    )


def construct_http_client_configuration_for_detection_step(
    step: ObjectDetectionModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> InferenceConfiguration:
    resolve = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    return InferenceConfiguration(
        disable_active_learning=resolve(step.disable_active_learning),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence_threshold=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
        max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
        max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
    )


def construct_http_client_configuration_for_segmentation_step(
    step: InstanceSegmentationModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> InferenceConfiguration:
    resolve = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    return InferenceConfiguration(
        disable_active_learning=resolve(step.disable_active_learning),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence_threshold=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
        mask_decode_mode=resolve(step.mask_decode_mode),
        tradeoff_factor=resolve(step.tradeoff_factor),
        max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
        max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
    )


def construct_http_client_configuration_for_keypoints_detection_step(
    step: KeypointsDetectionModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
) -> InferenceConfiguration:
    resolve = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    return InferenceConfiguration(
        disable_active_learning=resolve(step.disable_active_learning),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence_threshold=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
        keypoint_confidence_threshold=resolve(step.keypoint_confidence),
        max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
        max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
    )


MODEL_TYPE2HTTP_CLIENT_CONSTRUCTOR = {
    "ClassificationModel": construct_http_client_configuration_for_classification_step,
    "MultiLabelClassificationModel": construct_http_client_configuration_for_classification_step,
    "ObjectDetectionModel": construct_http_client_configuration_for_detection_step,
    "InstanceSegmentationModel": construct_http_client_configuration_for_segmentation_step,
    "KeypointsDetectionModel": construct_http_client_configuration_for_keypoints_detection_step,
}


async def run_yolo_world_model_step(
    step: YoloWorld,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
) -> Tuple[NextStepReference, OutputsLookup]:
    image = get_image(
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    class_names = resolve_parameter(
        selector_or_value=step.class_names,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    model_version = resolve_parameter(
        selector_or_value=step.version,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    confidence = resolve_parameter(
        selector_or_value=step.confidence,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    if step_execution_mode is StepExecutionMode.LOCAL:
        serialised_result = await get_yolo_world_predictions_locally(
            image=image,
            class_names=class_names,
            model_version=model_version,
            confidence=confidence,
            model_manager=model_manager,
            api_key=api_key,
        )
    else:
        serialised_result = await get_yolo_world_predictions_from_remote_api(
            image=image,
            class_names=class_names,
            model_version=model_version,
            confidence=confidence,
            step=step,
            api_key=api_key,
        )
    serialised_result = attach_prediction_type_info(
        results=serialised_result,
        prediction_type="object-detection",
    )
    serialised_result = attach_parent_info(image=image, results=serialised_result)
    serialised_result = anchor_detections_in_parent_coordinates(
        image=image,
        serialised_result=serialised_result,
    )
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


async def get_yolo_world_predictions_locally(
    image: List[dict],
    class_names: List[str],
    model_version: Optional[str],
    confidence: Optional[float],
    model_manager: ModelManager,
    api_key: Optional[str],
) -> List[dict]:
    serialised_result = []
    for single_image in image:
        inference_request = YOLOWorldInferenceRequest(
            image=single_image,
            yolo_world_version_id=model_version,
            confidence=confidence,
            text=class_names,
        )
        yolo_world_model_id = load_core_model(
            model_manager=model_manager,
            inference_request=inference_request,
            core_model="yolo_world",
            api_key=api_key,
        )
        result = await model_manager.infer_from_request(
            yolo_world_model_id, inference_request
        )
        serialised_result.append(result.dict())
    return serialised_result


async def get_yolo_world_predictions_from_remote_api(
    image: List[dict],
    class_names: List[str],
    model_version: Optional[str],
    confidence: Optional[float],
    step: YoloWorld,
    api_key: Optional[str],
) -> List[dict]:
    api_url = resolve_model_api_url(step=step)
    client = InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key,
    )
    configuration = InferenceConfiguration(
        max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
    )
    client.configure(inference_configuration=configuration)
    if WORKFLOWS_REMOTE_API_TARGET == "hosted":
        client.select_api_v0()
    image_batches = list(
        make_batches(
            iterable=image,
            batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
    )
    serialised_result = []
    for single_batch in image_batches:
        batch_results = await client.infer_from_yolo_world_async(
            inference_input=[i["value"] for i in single_batch],
            class_names=class_names,
            model_version=model_version,
            confidence=confidence,
        )
        serialised_result.extend(batch_results)
    return serialised_result


async def run_ocr_model_step(
    step: OCRModel,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
) -> Tuple[NextStepReference, OutputsLookup]:
    image = get_image(
        step=step,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    if step_execution_mode is StepExecutionMode.LOCAL:
        serialised_result = await get_ocr_predictions_locally(
            image=image,
            model_manager=model_manager,
            api_key=api_key,
        )
    else:
        serialised_result = await get_ocr_predictions_from_remote_api(
            step=step,
            image=image,
            api_key=api_key,
        )
    serialised_result = attach_parent_info(
        image=image,
        results=serialised_result,
        nested_key=None,
    )
    serialised_result = attach_prediction_type_info(
        results=serialised_result,
        prediction_type="ocr",
    )
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


async def get_ocr_predictions_locally(
    image: List[dict],
    model_manager: ModelManager,
    api_key: Optional[str],
) -> List[dict]:
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
    return serialised_result


async def get_ocr_predictions_from_remote_api(
    step: OCRModel,
    image: List[dict],
    api_key: Optional[str],
) -> List[dict]:
    api_url = resolve_model_api_url(step=step)
    client = InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key,
    )
    if WORKFLOWS_REMOTE_API_TARGET == "hosted":
        client.select_api_v0()
    configuration = InferenceConfiguration(
        max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
        max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
    )
    client.configure(configuration)
    result = await client.ocr_image_async(
        inference_input=[i["value"] for i in image],
    )
    if len(image) == 1:
        return [result]
    return result


async def run_clip_comparison_step(
    step: ClipComparison,
    runtime_parameters: Dict[str, Any],
    outputs_lookup: OutputsLookup,
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
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
    if step_execution_mode is StepExecutionMode.LOCAL:
        serialised_result = await get_clip_comparison_locally(
            image=image,
            text=text,
            model_manager=model_manager,
            api_key=api_key,
        )
    else:
        serialised_result = await get_clip_comparison_from_remote_api(
            step=step,
            image=image,
            text=text,
            api_key=api_key,
        )
    serialised_result = attach_parent_info(
        image=image,
        results=serialised_result,
        nested_key=None,
    )
    serialised_result = attach_prediction_type_info(
        results=serialised_result,
        prediction_type="embeddings-comparison",
    )
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


async def get_clip_comparison_locally(
    image: List[dict],
    text: str,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> List[dict]:
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
    return serialised_result


async def get_clip_comparison_from_remote_api(
    step: ClipComparison,
    image: List[dict],
    text: str,
    api_key: Optional[str],
) -> List[dict]:
    api_url = resolve_model_api_url(step=step)
    client = InferenceHTTPClient(
        api_url=api_url,
        api_key=api_key,
    )
    if WORKFLOWS_REMOTE_API_TARGET == "hosted":
        client.select_api_v0()
    image_batches = list(
        make_batches(
            iterable=image,
            batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
    )
    serialised_result = []
    for single_batch in image_batches:
        coroutines = []
        for single_image in single_batch:
            coroutine = client.clip_compare_async(
                subject=single_image["value"],
                prompt=text,
            )
            coroutines.append(coroutine)
        batch_results = list(await asyncio.gather(*coroutines))
        serialised_result.extend(batch_results)
    return serialised_result


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


def attach_prediction_type_info(
    results: List[Dict[str, Any]],
    prediction_type: str,
    key: str = "prediction_type",
) -> List[Dict[str, Any]]:
    for result in results:
        result[key] = prediction_type
    return results


def attach_parent_info(
    image: List[Dict[str, Any]],
    results: List[Dict[str, Any]],
    nested_key: Optional[str] = "predictions",
) -> List[Dict[str, Any]]:
    return [
        attach_parent_info_to_image_detections(
            image=i, predictions=p, nested_key=nested_key
        )
        for i, p in zip(image, results)
    ]


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
    image: List[Dict[str, Any]],
    serialised_result: List[Dict[str, Any]],
    image_metadata_key: str = "image",
    detections_key: str = "predictions",
) -> List[Dict[str, Any]]:
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


ROBOFLOW_MODEL2HOSTED_ENDPOINT = {
    "ClassificationModel": HOSTED_CLASSIFICATION_URL,
    "MultiLabelClassificationModel": HOSTED_CLASSIFICATION_URL,
    "ObjectDetectionModel": HOSTED_DETECT_URL,
    "KeypointsDetectionModel": HOSTED_DETECT_URL,
    "InstanceSegmentationModel": HOSTED_INSTANCE_SEGMENTATION_URL,
    "OCRModel": HOSTED_CORE_MODEL_URL,
    "ClipComparison": HOSTED_CORE_MODEL_URL,
}


def resolve_model_api_url(step: StepInterface) -> str:
    if WORKFLOWS_REMOTE_API_TARGET != "hosted":
        return LOCAL_INFERENCE_API_URL
    return ROBOFLOW_MODEL2HOSTED_ENDPOINT[step.get_type()]
