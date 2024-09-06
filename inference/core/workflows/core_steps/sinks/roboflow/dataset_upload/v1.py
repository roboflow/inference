"""
*****************************************************************
*                           WARNING!                            *
*****************************************************************
This module contains the utility functions used by
RoboflowDatasetUploadBlockV2.

We do not recommend making multiple blocks dependent on the same code,
but the change between v1 and v2 was basically the default value of
some parameter - hence we decided not to replicate the code.

If you need to modify this module beware that you may introduce
change to RoboflowDatasetUploadBlockV2! If that happens,
probably that's the time to disentangle those blocks and copy the
code.
"""

import hashlib
import json
import logging
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from functools import partial
from typing import List, Literal, Optional, Tuple, Type, Union
from uuid import uuid4

import supervision as sv
from fastapi import BackgroundTasks
from pydantic import ConfigDict, Field

from inference.core.active_learning.cache_operations import (
    return_strategy_credit,
    use_credit_of_matching_strategy,
)
from inference.core.active_learning.core import prepare_image_to_registration
from inference.core.active_learning.entities import (
    ImageDimensions,
    StrategyLimit,
    StrategyLimitType,
)
from inference.core.cache.base import BaseCache
from inference.core.roboflow_api import (
    annotate_image_at_roboflow,
    get_roboflow_workspace,
    register_image_at_roboflow,
)
from inference.core.workflows.core_steps.common.serializers import (
    serialise_sv_detections,
)
from inference.core.workflows.core_steps.common.utils import scale_sv_detections
from inference.core.workflows.execution_engine.constants import INFERENCE_ID_KEY
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    CLASSIFICATION_PREDICTION_KIND,
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    KEYPOINT_DETECTION_PREDICTION_KIND,
    OBJECT_DETECTION_PREDICTION_KIND,
    ROBOFLOW_PROJECT_KIND,
    STRING_KIND,
    ImageInputField,
    StepOutputImageSelector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.prototypes.block import (
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

SHORT_DESCRIPTION = "Save images and predictions in your Roboflow Dataset"

LONG_DESCRIPTION = """
Block let users save their images and predictions into Roboflow Dataset. Persisting data from
production environments helps iteratively building more robust models. 

Block provides configuration options to decide how data should be stored and what are the limits 
to be applied. We advice using this block in combination with rate limiter blocks to effectively 
collect data that the model struggle with.
"""

WORKSPACE_NAME_CACHE_EXPIRE = 900  # 15 min
TIMESTAMP_FORMAT = "%Y_%m_%d"
DUPLICATED_STATUS = "Duplicated image"
BatchCreationFrequency = Literal["never", "daily", "weekly", "monthly"]


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Roboflow Dataset Upload",
            "version": "v1",
            "short_description": SHORT_DESCRIPTION,
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "sink",
        }
    )
    type: Literal["roboflow_core/roboflow_dataset_upload@v1", "RoboflowDatasetUpload"]
    images: Union[WorkflowImageSelector, StepOutputImageSelector] = ImageInputField
    predictions: Optional[
        StepOutputSelector(
            kind=[
                OBJECT_DETECTION_PREDICTION_KIND,
                INSTANCE_SEGMENTATION_PREDICTION_KIND,
                KEYPOINT_DETECTION_PREDICTION_KIND,
                CLASSIFICATION_PREDICTION_KIND,
            ]
        )
    ] = Field(
        default=None,
        description="Reference q detection-like predictions",
        examples=["$steps.object_detection_model.predictions"],
    )
    target_project: Union[
        WorkflowParameterSelector(kind=[ROBOFLOW_PROJECT_KIND]), str
    ] = Field(
        description="name of Roboflow dataset / project to be used as target for collected data",
        examples=["my_dataset", "$inputs.target_al_dataset"],
    )
    usage_quota_name: str = Field(
        description="Unique name for Roboflow project pointed by `target_project` parameter, that identifies "
        "usage quota applied for this block.",
        examples=["quota-for-data-sampling-1"],
    )
    persist_predictions: bool = Field(
        default=True,
        description="Boolean flag to decide if predictions should be registered along with images",
        examples=[True, False],
    )
    minutely_usage_limit: int = Field(
        default=10,
        description="Maximum number of data registration requests per minute accounted in scope of "
        "single server or whole Roboflow platform, depending on context of usage.",
        examples=[10, 60],
    )
    hourly_usage_limit: int = Field(
        default=100,
        description="Maximum number of data registration requests per hour accounted in scope of "
        "single server or whole Roboflow platform, depending on context of usage.",
        examples=[10, 60],
    )
    daily_usage_limit: int = Field(
        default=1000,
        description="Maximum number of data registration requests per day accounted in scope of "
        "single server or whole Roboflow platform, depending on context of usage.",
        examples=[10, 60],
    )
    max_image_size: Tuple[int, int] = Field(
        default=(512, 512),
        description="Maximum size of the image to be registered - bigger images will be "
        "downsized preserving aspect ratio. Format of data: `(width, height)`",
        examples=[(512, 512), (1920, 1080)],
    )
    compression_level: int = Field(
        default=75,
        gt=0,
        le=100,
        description="Compression level for images registered",
        examples=[75],
    )
    registration_tags: List[
        Union[WorkflowParameterSelector(kind=[STRING_KIND]), str]
    ] = Field(
        default_factory=list,
        description="Tags to be attached to registered datapoints",
        examples=[["location-florida", "factory-name", "$inputs.dynamic_tag"]],
    )
    disable_sink: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = Field(
        default=False,
        description="boolean flag that can be also reference to input - to arbitrarily disable "
        "data collection for specific request",
        examples=[True, "$inputs.disable_active_learning"],
    )
    fire_and_forget: Union[bool, WorkflowParameterSelector(kind=[BOOLEAN_KIND])] = (
        Field(
            default=True,
            description="Boolean flag dictating if sink is supposed to be executed in the background, "
            "not waiting on status of registration before end of workflow run. Use `True` if best-effort "
            "registration is needed, use `False` while debugging and if error handling is needed",
            examples=[True],
        )
    )
    labeling_batch_prefix: Union[str, WorkflowParameterSelector(kind=[STRING_KIND])] = (
        Field(
            default="workflows_data_collector",
            description="Prefix of the name for labeling batches that will be registered in Roboflow app",
            examples=["my_labeling_batch_name"],
        )
    )
    labeling_batches_recreation_frequency: BatchCreationFrequency = Field(
        default="never",
        description="Frequency in which new labeling batches are created in Roboflow app. New batches "
        "are created with name prefix provided in `labeling_batch_prefix` in given time intervals."
        "Useful in organising labeling flow.",
        examples=["never", "daily"],
    )

    @classmethod
    def accepts_batch_input(cls) -> bool:
        return True

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="error_status", kind=[BOOLEAN_KIND]),
            OutputDefinition(name="message", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.0.0,<2.0.0"


class RoboflowDatasetUploadBlockV1(WorkflowBlock):

    def __init__(
        self,
        cache: BaseCache,
        api_key: Optional[str],
        background_tasks: Optional[BackgroundTasks],
        thread_pool_executor: Optional[ThreadPoolExecutor],
    ):
        self._cache = cache
        self._api_key = api_key
        self._background_tasks = background_tasks
        self._thread_pool_executor = thread_pool_executor

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return ["cache", "api_key", "background_tasks", "thread_pool_executor"]

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        images: Batch[WorkflowImageData],
        predictions: Optional[Batch[Union[sv.Detections, dict]]],
        target_project: str,
        usage_quota_name: str,
        minutely_usage_limit: int,
        persist_predictions: bool,
        hourly_usage_limit: int,
        daily_usage_limit: int,
        max_image_size: Tuple[int, int],
        compression_level: int,
        registration_tags: List[str],
        disable_sink: bool,
        fire_and_forget: bool,
        labeling_batch_prefix: str,
        labeling_batches_recreation_frequency: BatchCreationFrequency,
    ) -> BlockResult:
        if self._api_key is None:
            raise ValueError(
                "RoboflowDataCollector block cannot run without Roboflow API key. "
                "If you do not know how to get API key - visit "
                "https://docs.roboflow.com/api-reference/authentication#retrieve-an-api-key to learn how to "
                "retrieve one."
            )
        if disable_sink:
            return [
                {
                    "error_status": False,
                    "message": "Sink was disabled by parameter `disable_sink`",
                }
                for _ in range(len(images))
            ]
        result = []
        predictions = [None] * len(images) if predictions is None else predictions
        for image, prediction in zip(images, predictions):
            error_status, message = register_datapoint_at_roboflow(
                image=image,
                prediction=prediction,
                target_project=target_project,
                usage_quota_name=usage_quota_name,
                persist_predictions=persist_predictions,
                minutely_usage_limit=minutely_usage_limit,
                hourly_usage_limit=hourly_usage_limit,
                daily_usage_limit=daily_usage_limit,
                max_image_size=max_image_size,
                compression_level=compression_level,
                registration_tags=registration_tags,
                fire_and_forget=fire_and_forget,
                labeling_batch_prefix=labeling_batch_prefix,
                new_labeling_batch_frequency=labeling_batches_recreation_frequency,
                cache=self._cache,
                background_tasks=self._background_tasks,
                thread_pool_executor=self._thread_pool_executor,
                api_key=self._api_key,
            )
            result.append({"error_status": error_status, "message": message})
        return result


def register_datapoint_at_roboflow(
    image: WorkflowImageData,
    prediction: Optional[Union[sv.Detections, dict]],
    target_project: str,
    usage_quota_name: str,
    persist_predictions: bool,
    minutely_usage_limit: int,
    hourly_usage_limit: int,
    daily_usage_limit: int,
    max_image_size: Tuple[int, int],
    compression_level: int,
    registration_tags: List[str],
    fire_and_forget: bool,
    labeling_batch_prefix: str,
    new_labeling_batch_frequency: BatchCreationFrequency,
    cache: BaseCache,
    background_tasks: Optional[BackgroundTasks],
    thread_pool_executor: Optional[ThreadPoolExecutor],
    api_key: str,
) -> Tuple[bool, str]:
    registration_task = partial(
        execute_registration,
        image=image,
        prediction=prediction,
        target_project=target_project,
        usage_quota_name=usage_quota_name,
        persist_predictions=persist_predictions,
        minutely_usage_limit=minutely_usage_limit,
        hourly_usage_limit=hourly_usage_limit,
        daily_usage_limit=daily_usage_limit,
        max_image_size=max_image_size,
        compression_level=compression_level,
        registration_tags=registration_tags,
        labeling_batch_prefix=labeling_batch_prefix,
        new_labeling_batch_frequency=new_labeling_batch_frequency,
        cache=cache,
        api_key=api_key,
    )
    if fire_and_forget and background_tasks:
        background_tasks.add_task(registration_task)
        return False, "Element registration happens in the background task"
    if fire_and_forget and thread_pool_executor:
        thread_pool_executor.submit(registration_task)
        return False, "Element registration happens in the background task"
    return registration_task()


def execute_registration(
    image: WorkflowImageData,
    prediction: Optional[Union[sv.Detections, dict]],
    target_project: str,
    persist_predictions: bool,
    usage_quota_name: str,
    minutely_usage_limit: int,
    hourly_usage_limit: int,
    daily_usage_limit: int,
    max_image_size: Tuple[int, int],
    compression_level: int,
    registration_tags: List[str],
    labeling_batch_prefix: str,
    new_labeling_batch_frequency: BatchCreationFrequency,
    cache: BaseCache,
    api_key: str,
) -> Tuple[bool, str]:
    matching_strategies_limits = OrderedDict(
        {
            usage_quota_name: [
                StrategyLimit(
                    limit_type=StrategyLimitType.MINUTELY, value=minutely_usage_limit
                ),
                StrategyLimit(
                    limit_type=StrategyLimitType.HOURLY, value=hourly_usage_limit
                ),
                StrategyLimit(
                    limit_type=StrategyLimitType.DAILY, value=daily_usage_limit
                ),
            ]
        }
    )
    workspace_name = get_workspace_name(api_key=api_key, cache=cache)
    strategy_with_spare_credit = use_credit_of_matching_strategy(
        cache=cache,
        workspace=workspace_name,
        project=target_project,
        matching_strategies_limits=matching_strategies_limits,
    )
    if strategy_with_spare_credit is None:
        return False, "Registration skipped due to usage quota exceeded"
    credit_to_be_returned = False
    try:
        local_image_id = str(uuid4())
        encoded_image, scaling_factor = prepare_image_to_registration(
            image=image.numpy_image,
            desired_size=ImageDimensions(
                width=max_image_size[0], height=max_image_size[1]
            ),
            jpeg_compression_level=compression_level,
        )
        batch_name = generate_batch_name(
            labeling_batch_prefix=labeling_batch_prefix,
            new_labeling_batch_frequency=new_labeling_batch_frequency,
        )
        if isinstance(prediction, sv.Detections):
            prediction = scale_sv_detections(
                detections=prediction, scale=scaling_factor
            )
        status = register_datapoint(
            target_project=target_project,
            encoded_image=encoded_image,
            local_image_id=local_image_id,
            prediction=prediction if persist_predictions else None,
            api_key=api_key,
            batch_name=batch_name,
            tags=registration_tags,
        )
        if status == DUPLICATED_STATUS:
            credit_to_be_returned = True
        return False, status
    except Exception as error:
        credit_to_be_returned = True
        logging.exception("Failed to register datapoint on the Roboflow platform")
        return (
            True,
            f"Error while registration. Error type: {type(error)}. Details: {error}",
        )
    finally:
        if credit_to_be_returned:
            return_strategy_credit(
                cache=cache,
                workspace=workspace_name,
                project=target_project,
                strategy_name=strategy_with_spare_credit,
            )


def get_workspace_name(
    api_key: str,
    cache: BaseCache,
) -> str:
    api_key_hash = hashlib.md5(api_key.encode("utf-8")).hexdigest()
    cache_key = f"workflows:api_key_to_workspace:{api_key_hash}"
    cached_workspace_name = cache.get(cache_key)
    if cached_workspace_name:
        return cached_workspace_name
    workspace_name_from_api = get_roboflow_workspace(api_key=api_key)
    cache.set(
        key=cache_key, value=workspace_name_from_api, expire=WORKSPACE_NAME_CACHE_EXPIRE
    )
    return workspace_name_from_api


def generate_batch_name(
    labeling_batch_prefix: str,
    new_labeling_batch_frequency: BatchCreationFrequency,
) -> str:
    if new_labeling_batch_frequency == "never":
        return labeling_batch_prefix
    timestamp_generator = RECREATION_INTERVAL2TIMESTAMP_GENERATOR[
        new_labeling_batch_frequency
    ]
    timestamp = timestamp_generator()
    return f"{labeling_batch_prefix}_{timestamp}"


def generate_today_timestamp() -> str:
    return datetime.today().strftime(TIMESTAMP_FORMAT)


def generate_start_timestamp_for_this_week() -> str:
    today = datetime.today()
    return (today - timedelta(days=today.weekday())).strftime(TIMESTAMP_FORMAT)


def generate_start_timestamp_for_this_month() -> str:
    return datetime.today().replace(day=1).strftime(TIMESTAMP_FORMAT)


RECREATION_INTERVAL2TIMESTAMP_GENERATOR = {
    "daily": generate_today_timestamp,
    "weekly": generate_start_timestamp_for_this_week,
    "monthly": generate_start_timestamp_for_this_month,
}


def register_datapoint(
    target_project: str,
    encoded_image: bytes,
    local_image_id: str,
    prediction: Optional[Union[sv.Detections, dict]],
    api_key: str,
    batch_name: str,
    tags: List[str],
) -> str:
    inference_id = None
    if isinstance(prediction, dict):
        inference_id = prediction.get(INFERENCE_ID_KEY)
    if isinstance(prediction, sv.Detections) and len(prediction) > 0:
        # TODO: Lack of inference ID for empty prediction -
        #  dependent on https://github.com/roboflow/inference/issues/567
        inference_id_array = prediction.data.get(INFERENCE_ID_KEY)
        if inference_id_array is not None:
            inference_id_list = inference_id_array.tolist()
            inference_id = inference_id_list[0]
    roboflow_image_id = safe_register_image_at_roboflow(
        target_project=target_project,
        encoded_image=encoded_image,
        local_image_id=local_image_id,
        api_key=api_key,
        batch_name=batch_name,
        tags=tags,
        inference_id=inference_id,
    )
    if roboflow_image_id is None:
        return DUPLICATED_STATUS
    if is_prediction_registration_forbidden(prediction=prediction):
        return "Successfully registered image"
    encoded_prediction, prediction_format = encode_prediction(prediction=prediction)
    _ = annotate_image_at_roboflow(
        api_key=api_key,
        dataset_id=target_project,
        local_image_id=local_image_id,
        roboflow_image_id=roboflow_image_id,
        annotation_content=encoded_prediction,
        annotation_file_type=prediction_format,
        is_prediction=True,
    )
    return "Successfully registered image and annotation"


def safe_register_image_at_roboflow(
    target_project: str,
    encoded_image: bytes,
    local_image_id: str,
    api_key: str,
    batch_name: str,
    tags: List[str],
    inference_id: Optional[str],
) -> Optional[str]:
    registration_response = register_image_at_roboflow(
        api_key=api_key,
        dataset_id=target_project,
        local_image_id=local_image_id,
        image_bytes=encoded_image,
        batch_name=batch_name,
        tags=tags,
        inference_id=inference_id,
    )
    image_duplicated = registration_response.get("duplicate", False)
    if image_duplicated:
        logging.warning(f"Image duplication detected: {registration_response}.")
        return None
    return registration_response["id"]


def is_prediction_registration_forbidden(
    prediction: Optional[Union[sv.Detections, dict]],
) -> bool:
    if prediction is None:
        return True
    if isinstance(prediction, sv.Detections) and len(prediction) == 0:
        return True
    if isinstance(prediction, dict) and "top" not in prediction:
        return True
    return False


def encode_prediction(
    prediction: Union[sv.Detections, dict],
) -> Tuple[str, str]:
    if isinstance(prediction, dict):
        return prediction["top"], "txt"
    detections_in_inference_format = serialise_sv_detections(detections=prediction)
    return json.dumps(detections_in_inference_format), "json"
