from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field

from inference.core.interfaces.camera.video_source import (
    BufferConsumptionStrategy,
    BufferFillingStrategy,
)

STATUS_KEY = "status"
TYPE_KEY = "type"
ERROR_TYPE_KEY = "error_type"
REQUEST_ID_KEY = "request_id"
PIPELINE_ID_KEY = "pipeline_id"
COMMAND_KEY = "command"
RESPONSE_KEY = "response"
ENCODING = "utf-8"


class OperationStatus(str, Enum):
    SUCCESS = "success"
    FAILURE = "failure"


class ErrorType(str, Enum):
    INTERNAL_ERROR = "internal_error"
    INVALID_PAYLOAD = "invalid_payload"
    NOT_FOUND = "not_found"
    OPERATION_ERROR = "operation_error"
    AUTHORISATION_ERROR = "authorisation_error"


class CommandType(str, Enum):
    INIT = "init"
    MUTE = "mute"
    RESUME = "resume"
    STATUS = "status"
    TERMINATE = "terminate"
    LIST_PIPELINES = "list_pipelines"
    CONSUME_RESULT = "consume_result"


class VideoConfiguration(BaseModel):
    type: Literal["VideoConfiguration"]
    video_reference: Union[str, int, List[Union[str, int]]]
    max_fps: Optional[Union[float, int]] = None
    source_buffer_filling_strategy: Optional[BufferFillingStrategy] = (
        BufferFillingStrategy.DROP_OLDEST
    )
    source_buffer_consumption_strategy: Optional[BufferConsumptionStrategy] = (
        BufferConsumptionStrategy.EAGER
    )
    video_source_properties: Optional[Dict[str, float]] = None
    batch_collection_timeout: Optional[float] = None


class MemorySinkConfiguration(BaseModel):
    type: Literal["MemorySinkConfiguration"]
    results_buffer_size: int = 64


class WorkflowConfiguration(BaseModel):
    type: Literal["WorkflowConfiguration"]
    workflow_specification: Optional[dict] = None
    workspace_name: Optional[str] = None
    workflow_id: Optional[str] = None
    image_input_name: str = "image"
    workflows_parameters: Optional[Dict[str, Any]] = None
    workflows_thread_pool_workers: int = 4
    cancel_thread_pool_tasks_on_exit: bool = True
    video_metadata_input_name: str = "video_metadata"


class InitialisePipelinePayload(BaseModel):
    video_configuration: VideoConfiguration
    processing_configuration: WorkflowConfiguration
    sink_configuration: MemorySinkConfiguration = MemorySinkConfiguration(
        type="MemorySinkConfiguration"
    )
    api_key: Optional[str] = None


class ConsumeResultsPayload(BaseModel):
    excluded_fields: List[str] = Field(
        default_factory=list,
        description="List of workflow output fields to be filtered out from response",
    )
