import asyncio
import base64
import json
import math
import re
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from uuid import uuid4

import cv2
import numpy as np
import zxingcpp
from openai import AsyncOpenAI

from inference.core.entities.requests.clip import ClipCompareRequest
from inference.core.entities.requests.cogvlm import CogVLMInferenceRequest
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
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes, load_image
from inference.enterprise.workflows.complier.entities import StepExecutionMode
from inference.enterprise.workflows.complier.steps_executors.constants import (
    CENTER_X_KEY,
    CENTER_Y_KEY,
    HEIGHT_KEY,
    ORIGIN_COORDINATES_KEY,
    ORIGIN_SIZE_KEY,
    PARENT_COORDINATES_SUFFIX,
    WIDTH_KEY,
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
    GPT_4V_MODEL_TYPE,
    LMM,
    BarcodeDetection,
    ClassificationModel,
    ClipComparison,
    InstanceSegmentationModel,
    KeypointsDetectionModel,
    LMMConfig,
    LMMForClassification,
    MultiLabelClassificationModel,
    ObjectDetectionModel,
    OCRModel,
    QRCodeDetection,
    RoboflowModel,
    StepInterface,
    YoloWorld,
)
from inference.enterprise.workflows.errors import ExecutionGraphError
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

MODEL_TYPE2PREDICTION_TYPE = {
    "ClassificationModel": "classification",
    "MultiLabelClassificationModel": "classification",
    "ObjectDetectionModel": "object-detection",
    "InstanceSegmentationModel": "instance-segmentation",
    "KeypointsDetectionModel": "keypoint-detection",
}

NOT_DETECTED_VALUE = "not_detected"
JSON_MARKDOWN_BLOCK_PATTERN = re.compile(r"```json\n([\s\S]*?)\n```")


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
        class_filter = resolve_parameter(
            selector_or_value=step.class_filter,
            runtime_parameters=runtime_parameters,
            outputs_lookup=outputs_lookup,
        )
        # we need to filter-out classes here, to ensure consistent behaviour of external API usage
        # as legacy endpoints to not support this functionality
        serialised_result = filter_out_unwanted_classes(
            serialised_result=serialised_result,
            classes_to_accept=class_filter,
        )
        serialised_result = attach_parent_info(image=image, results=serialised_result)
        serialised_result = anchor_detections_in_parent_coordinates(
            image=image,
            serialised_result=serialised_result,
        )
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


def filter_out_unwanted_classes(
    serialised_result: List[Dict[str, Any]],
    classes_to_accept: Optional[List[str]],
) -> List[Dict[str, Any]]:
    if classes_to_accept is None:
        return serialised_result
    classes_to_accept = set(classes_to_accept)
    results = []
    for image_result in serialised_result:
        filtered_image_result = deepcopy(image_result)
        filtered_image_result["predictions"] = []
        for prediction in image_result["predictions"]:
            if prediction["class"] not in classes_to_accept:
                continue
            filtered_image_result["predictions"].append(prediction)
        results.append(filtered_image_result)
    return results


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
        source="workflow-execution",
        active_learning_target_dataset=resolve(step.active_learning_target_dataset),
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
        active_learning_target_dataset=resolve(step.active_learning_target_dataset),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
        source="workflow-execution",
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
        active_learning_target_dataset=resolve(step.active_learning_target_dataset),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
        mask_decode_mode=resolve(step.mask_decode_mode),
        tradeoff_factor=resolve(step.tradeoff_factor),
        source="workflow-execution",
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
        active_learning_target_dataset=resolve(step.active_learning_target_dataset),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
        keypoint_confidence=resolve(step.keypoint_confidence),
        source="workflow-execution",
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
        active_learning_target_dataset=resolve(step.active_learning_target_dataset),
        max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
        max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        source="workflow-execution",
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
        active_learning_target_dataset=resolve(step.active_learning_target_dataset),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence_threshold=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
        max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
        max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        source="workflow-execution",
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
        active_learning_target_dataset=resolve(step.active_learning_target_dataset),
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
        source="workflow-execution",
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
        active_learning_target_dataset=resolve(step.active_learning_target_dataset),
        class_agnostic_nms=resolve(step.class_agnostic_nms),
        class_filter=resolve(step.class_filter),
        confidence_threshold=resolve(step.confidence),
        iou_threshold=resolve(step.iou_threshold),
        max_detections=resolve(step.max_detections),
        max_candidates=resolve(step.max_candidates),
        keypoint_confidence_threshold=resolve(step.keypoint_confidence),
        max_batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_BATCH_SIZE,
        max_concurrent_requests=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        source="workflow-execution",
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
        serialised_result.append(result.dict(by_alias=True, exclude_none=True))
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
    parent_left_top_x = round(shift_x - image[ORIGIN_COORDINATES_KEY][WIDTH_KEY] / 2)
    parent_left_top_y = round(shift_y - image[ORIGIN_COORDINATES_KEY][HEIGHT_KEY] / 2)
    for detection in serialised_result[f"{detections_key}{PARENT_COORDINATES_SUFFIX}"]:
        detection["x"] += parent_left_top_x
        detection["y"] += parent_left_top_y
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


async def run_lmm_step(
    step: LMM,
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
    resolve_parameter_closure = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    prompt = resolve_parameter_closure(step.prompt)
    lmm_type = resolve_parameter_closure(step.lmm_type)
    remote_api_key = resolve_parameter_closure(step.remote_api_key)
    json_output = resolve_parameter_closure(step.json_output)
    if json_output is not None:
        prompt = (
            f"{prompt}\n\nVALID response format is JSON:\n"
            f"{json.dumps(json_output, indent=4)}"
        )
    if lmm_type == GPT_4V_MODEL_TYPE:
        raw_output, structured_output = await run_gpt_4v_llm_prompting(
            image=image,
            prompt=prompt,
            remote_api_key=remote_api_key,
            lmm_config=step.lmm_config,
            expected_output=json_output,
        )
    else:
        raw_output, structured_output = await run_cog_vlm_prompting(
            image=image,
            prompt=prompt,
            expected_output=json_output,
            model_manager=model_manager,
            api_key=api_key,
            step_execution_mode=step_execution_mode,
        )
    serialised_result = [
        {
            "raw_output": raw["content"],
            "image": raw["image"],
            "structured_output": structured,
            **structured,
        }
        for raw, structured in zip(raw_output, structured_output)
    ]
    serialised_result = attach_parent_info(
        image=image,
        results=serialised_result,
        nested_key=None,
    )
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


async def run_lmm_for_classification_step(
    step: LMMForClassification,
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
    resolve_parameter_closure = partial(
        resolve_parameter,
        runtime_parameters=runtime_parameters,
        outputs_lookup=outputs_lookup,
    )
    lmm_type = resolve_parameter_closure(step.lmm_type)
    remote_api_key = resolve_parameter_closure(step.remote_api_key)
    classes = resolve_parameter_closure(step.classes)
    prompt = (
        f"You are supposed to perform image classification task. You are given image that should be "
        f"assigned one of the following classes: {classes}. "
        f'Your response must be JSON in format: {{"top": "some_class"}}'
    )
    if lmm_type == GPT_4V_MODEL_TYPE:
        raw_output, structured_output = await run_gpt_4v_llm_prompting(
            image=image,
            prompt=prompt,
            remote_api_key=remote_api_key,
            lmm_config=step.lmm_config,
            expected_output={"top": "name of the class"},
        )
    else:
        raw_output, structured_output = await run_cog_vlm_prompting(
            image=image,
            prompt=prompt,
            expected_output={"top": "name of the class"},
            model_manager=model_manager,
            api_key=api_key,
            step_execution_mode=step_execution_mode,
        )
    serialised_result = [
        {"raw_output": raw["content"], "image": raw["image"], "top": structured["top"]}
        for raw, structured in zip(raw_output, structured_output)
    ]
    serialised_result = attach_parent_info(
        image=image,
        results=serialised_result,
        nested_key=None,
    )
    serialised_result = attach_prediction_type_info(
        results=serialised_result,
        prediction_type="classification",
    )
    outputs_lookup[construct_step_selector(step_name=step.name)] = serialised_result
    return None, outputs_lookup


async def run_gpt_4v_llm_prompting(
    image: List[Dict[str, Any]],
    prompt: str,
    remote_api_key: Optional[str],
    lmm_config: LMMConfig,
    expected_output: Optional[Dict[str, str]],
) -> Tuple[List[Dict[str, str]], List[dict]]:
    if remote_api_key is None:
        raise ExecutionGraphError(
            f"Step that involves GPT-4V prompting requires OpenAI API key which was not provided."
        )
    results = await execute_gpt_4v_requests(
        image=image,
        remote_api_key=remote_api_key,
        prompt=prompt,
        lmm_config=lmm_config,
    )
    if expected_output is None:
        return results, [{} for _ in range(len(results))]
    parsed_output = [
        try_parse_lmm_output_to_json(
            output=r["content"], expected_output=expected_output
        )
        for r in results
    ]
    return results, parsed_output


async def execute_gpt_4v_requests(
    image: List[dict],
    remote_api_key: str,
    prompt: str,
    lmm_config: LMMConfig,
) -> List[Dict[str, str]]:
    client = AsyncOpenAI(api_key=remote_api_key)
    results = []
    images_batches = list(
        make_batches(
            iterable=image,
            batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
    )
    for image_batch in images_batches:
        batch_coroutines = []
        for image in image_batch:
            coroutine = execute_gpt_4v_request(
                client=client,
                image=image,
                prompt=prompt,
                lmm_config=lmm_config,
            )
            batch_coroutines.append(coroutine)
        batch_results = await asyncio.gather(*batch_coroutines)
        results.extend(batch_results)
    return results


async def execute_gpt_4v_request(
    client: AsyncOpenAI,
    image: Dict[str, Any],
    prompt: str,
    lmm_config: LMMConfig,
) -> Dict[str, str]:
    loaded_image, _ = load_image(image)
    image_metadata = {"width": loaded_image.shape[1], "height": loaded_image.shape[0]}
    base64_image = base64.b64encode(encode_image_to_jpeg_bytes(loaded_image)).decode(
        "ascii"
    )
    response = await client.chat.completions.create(
        model=lmm_config.gpt_model_version,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                            "detail": lmm_config.gpt_image_detail,
                        },
                    },
                ],
            }
        ],
        max_tokens=lmm_config.max_tokens,
    )
    return {"content": response.choices[0].message.content, "image": image_metadata}


async def run_cog_vlm_prompting(
    image: List[Dict[str, Any]],
    prompt: str,
    expected_output: Optional[Dict[str, str]],
    model_manager: ModelManager,
    api_key: Optional[str],
    step_execution_mode: StepExecutionMode,
) -> Tuple[List[Dict[str, Any]], List[dict]]:
    if step_execution_mode is StepExecutionMode.LOCAL:
        cogvlm_generations = await get_cogvlm_generations_locally(
            image=image,
            prompt=prompt,
            model_manager=model_manager,
            api_key=api_key,
        )
    else:
        cogvlm_generations = await get_cogvlm_generations_from_remote_api(
            image=image,
            prompt=prompt,
            api_key=api_key,
        )
    if expected_output is None:
        return cogvlm_generations, [{} for _ in range(len(cogvlm_generations))]
    parsed_output = [
        try_parse_lmm_output_to_json(
            output=r["content"],
            expected_output=expected_output,
        )
        for r in cogvlm_generations
    ]
    return cogvlm_generations, parsed_output


async def get_cogvlm_generations_locally(
    image: List[dict],
    prompt: str,
    model_manager: ModelManager,
    api_key: Optional[str],
) -> List[Dict[str, Any]]:
    serialised_result = []
    for single_image in image:
        loaded_image, _ = load_image(single_image)
        image_metadata = {
            "width": loaded_image.shape[1],
            "height": loaded_image.shape[0],
        }
        inference_request = CogVLMInferenceRequest(
            image=single_image,
            prompt=prompt,
        )
        yolo_world_model_id = load_core_model(
            model_manager=model_manager,
            inference_request=inference_request,
            core_model="cogvlm",
            api_key=api_key,
        )
        result = await model_manager.infer_from_request(
            yolo_world_model_id, inference_request
        )
        serialised_result.append(
            {
                "content": result.response,
                "image": image_metadata,
            }
        )
    return serialised_result


async def get_cogvlm_generations_from_remote_api(
    image: List[dict],
    prompt: str,
    api_key: Optional[str],
) -> List[Dict[str, Any]]:
    if WORKFLOWS_REMOTE_API_TARGET == "hosted":
        raise ExecutionGraphError(
            f"Chosen remote execution of CogVLM model in Roboflow Hosted API mode, but remote execution "
            f"is only possible for self-hosted option."
        )
    client = InferenceHTTPClient.init(
        api_url=LOCAL_INFERENCE_API_URL,
        api_key=api_key,
    )
    results = []
    images_batches = list(
        make_batches(
            iterable=image,
            batch_size=WORKFLOWS_REMOTE_EXECUTION_MAX_STEP_CONCURRENT_REQUESTS,
        )
    )
    for image_batch in images_batches:
        batch_coroutines, batch_image_metadata = [], []
        for image in image_batch:
            loaded_image, _ = load_image(image)
            image_metadata = {
                "width": loaded_image.shape[1],
                "height": loaded_image.shape[0],
            }
            batch_image_metadata.append(image_metadata)
            coroutine = client.prompt_cogvlm_async(
                visual_prompt=image["value"],
                text_prompt=prompt,
            )
            batch_coroutines.append(coroutine)
        batch_results = await asyncio.gather(*batch_coroutines)
        results.extend(
            [
                {"content": br["response"], "image": bm}
                for br, bm in zip(batch_results, batch_image_metadata)
            ]
        )
    return results


def try_parse_lmm_output_to_json(
    output: str, expected_output: Dict[str, str]
) -> Union[list, dict]:
    json_blocks_found = JSON_MARKDOWN_BLOCK_PATTERN.findall(output)
    if len(json_blocks_found) == 0:
        return try_parse_json(output, expected_output=expected_output)
    result = []
    for json_block in json_blocks_found:
        result.append(
            try_parse_json(content=json_block, expected_output=expected_output)
        )
    return result if len(result) > 1 else result[0]


def try_parse_json(content: str, expected_output: Dict[str, str]) -> dict:
    try:
        data = json.loads(content)
        return {key: data.get(key, NOT_DETECTED_VALUE) for key in expected_output}
    except Exception:
        return {key: NOT_DETECTED_VALUE for key in expected_output}


def resolve_model_api_url(step: StepInterface) -> str:
    if WORKFLOWS_REMOTE_API_TARGET != "hosted":
        return LOCAL_INFERENCE_API_URL
    return ROBOFLOW_MODEL2HOSTED_ENDPOINT[step.get_type()]


async def run_qr_code_detection_step(
    step: QRCodeDetection,
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
    decoded_images = [load_image(e)[0] for e in image]
    image_metadata = [
        {"width": img.shape[1], "height": img.shape[0]} for img in decoded_images
    ]
    image_parent_ids = [img["parent_id"] for img in image]
    predictions = [
        detect_qr_codes(image=image, parent_id=parent_id)
        for image, parent_id in zip(decoded_images, image_parent_ids)
    ]

    outputs_lookup[construct_step_selector(step_name=step.name)] = {
        "parent_id": image_parent_ids,
        "predictions": predictions,
        "image": image_metadata,
        "prediction_type": "qrcode-detection",
    }
    return None, outputs_lookup


def detect_qr_codes(
    image: np.ndarray, parent_id: str
) -> Dict[str, Union[str, np.ndarray]]:
    detector = cv2.QRCodeDetector()
    retval, detections, pointsList, _ = detector.detectAndDecodeMulti(image)
    predictions = []
    for data, points in zip(detections, pointsList):
        width = points[2][0] - points[0][0]
        height = points[2][1] - points[0][1]
        predictions.append(
            {
                "parent_id": parent_id,
                "class": "qr_code",
                "class_id": 0,
                "confidence": 1.0,
                "x": points[0][0] + width / 2,
                "y": points[0][1] + height / 2,
                "width": width,
                "height": height,
                "detection_id": str(uuid4()),
                "data": data,
            }
        )
    return predictions


async def run_barcode_detection_step(
    step: BarcodeDetection,
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
    decoded_images = [load_image(e)[0] for e in image]
    image_metadata = [
        {"width": img.shape[1], "height": img.shape[0]} for img in decoded_images
    ]
    image_parent_ids = [img["parent_id"] for img in image]
    predictions = [
        detect_barcodes(image=image, parent_id=parent_id)
        for image, parent_id in zip(decoded_images, image_parent_ids)
    ]

    outputs_lookup[construct_step_selector(step_name=step.name)] = {
        "parent_id": image_parent_ids,
        "predictions": predictions,
        "image": image_metadata,
        "prediction_type": "barcode-detection",
    }
    return None, outputs_lookup


def detect_barcodes(
    image: np.ndarray, parent_id: str
) -> Dict[str, Union[str, np.ndarray]]:
    barcodes = zxingcpp.read_barcodes(image)
    predictions = []

    for barcode in barcodes:
        width = barcode.position.top_right.x - barcode.position.top_left.x
        height = barcode.position.bottom_left.y - barcode.position.top_left.y

        predictions.append(
            {
                "parent_id": parent_id,
                "class": "barcode",
                "class_id": 0,
                "confidence": 1.0,
                "x": int(math.floor(barcode.position.top_left.x + width / 2)),
                "y": int(math.floor(barcode.position.top_left.y + height / 2)),
                "width": width,
                "height": height,
                "detection_id": str(uuid4()),
                "data": barcode.text,
            }
        )
    return predictions
