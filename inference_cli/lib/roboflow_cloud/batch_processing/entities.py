from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field


class JobMetadata(BaseModel):
    job_id: str = Field(alias="jobId")
    name: str
    job_definition: dict = Field(alias="jobDefinition")
    current_stage: Optional[str] = Field(alias="currentStage", default=None)
    planned_stages: Optional[List[str]] = Field(alias="plannedStages", default=None)
    error: bool = Field(default=False)
    is_terminal: bool = Field(alias="isTerminal", default=False)
    last_notification: Optional[Union[dict, str]] = Field(
        alias="lastNotification", default=None
    )
    created_at: datetime = Field(alias="createdAt")
    last_update: datetime = Field(alias="lastUpdate")
    restart_parameters_override: List[dict] = Field(
        alias="restartParametersOverride", default_factory=list
    )


class ListBatchJobsResponse(BaseModel):
    jobs: List[JobMetadata]
    next_page_token: Optional[str] = Field(alias="nextPageToken")


class GetJobMetadataResponse(BaseModel):
    job: JobMetadata


class JobStageDetails(BaseModel):
    processing_stage_id: str = Field(alias="processingStageId")
    processing_stage_name: str = Field(alias="processingStageName")
    tasks_number: int = Field(alias="tasksNumber")
    output_batches: List[str] = Field(alias="outputBatches")
    start_timestamp: datetime = Field(alias="startTimestamp")
    notification: Union[dict, str] = Field(alias="notification")
    status_type: str = Field(alias="statusType")
    last_event_timestamp: datetime = Field(alias="lastEventTimestamp")


class ListJobStagesResponse(BaseModel):
    stages: List[JobStageDetails]


class TaskStatus(BaseModel):
    task_id: str = Field(alias="taskId")
    notification: Union[dict, str] = Field(alias="notification")
    status_type: str = Field(alias="statusType")
    progress: Optional[float] = Field(default=None)
    event_timestamp: datetime = Field(alias="eventTimestamp")


class ListJobStageTasksResponse(BaseModel):
    tasks: List[TaskStatus]
    next_page_token: Optional[str] = Field(alias="nextPageToken")


class WorkflowProcessingJobType(str, Enum):
    JOB_TYPE_SIMPLE_IMAGE_PROCESSING_V1 = "simple-image-processing-v1"
    JOB_TYPE_SIMPLE_VIDEO_PROCESSING_V1 = "simple-video-processing-v1"


class MachineType(str, Enum):
    CPU = "cpu"
    GPU = "gpu"


class MachineSize(str, Enum):
    XS = "xs"
    S = "s"
    M = "m"
    L = "l"
    XL = "xl"


class ComputeConfigurationV2(BaseModel):
    type: Literal["compute-configuration-v2"] = Field(
        default="compute-configuration-v2"
    )
    machine_type: Optional[MachineType] = Field(
        serialization_alias="machineType", default=None
    )
    workers_per_machine: Optional[int] = Field(
        serialization_alias="workersPerMachine", default=None
    )


class StagingBatchInputV1(BaseModel):
    type: Literal["staging-batch-input-v1"] = Field(default="staging-batch-input-v1")
    batch_id: str = Field(serialization_alias="batchId")
    part_name: Optional[str] = Field(serialization_alias="partName", default=None)


class AggregationFormat(str, Enum):
    CSV = "csv"
    JSONL = "jsonl"


class WorkflowsProcessingSpecificationV1(BaseModel):
    type: Literal["workflows-processing-specification-v1"] = Field(
        default="workflows-processing-specification-v1"
    )
    workspace: str
    workflow_id: str = Field(serialization_alias="workflowId")
    workflow_parameters: Optional[Dict[str, Any]] = Field(
        serialization_alias="workflowParameters", default=None
    )
    image_input_name: Optional[str] = Field(
        serialization_alias="imageInputName", default=None
    )
    persist_images_outputs: Optional[bool] = Field(
        serialization_alias="persistImagesOutputs", default=None
    )
    images_outputs_to_be_persisted: Optional[List[str]] = Field(
        serialization_alias="imagesOutputsToBePersisted", default=None
    )
    aggregation_format: Optional[AggregationFormat] = Field(
        serialization_alias="aggregationFormat", default=None
    )
    max_video_fps: Optional[Union[int, float]] = Field(
        serialization_alias="maxVideoFPS", default=None
    )


class WorkflowProcessingJobV1(BaseModel):
    type: WorkflowProcessingJobType
    job_input: StagingBatchInputV1 = Field(serialization_alias="jobInput")
    compute_configuration: ComputeConfigurationV2 = Field(
        serialization_alias="computeConfiguration"
    )
    processing_timeout_seconds: Optional[int] = Field(
        serialization_alias="processingTimeoutSeconds", default=None
    )
    max_parallel_tasks: Optional[int] = Field(
        serialization_alias="maxParallelTasks", default=None
    )
    processing_specification: WorkflowsProcessingSpecificationV1 = Field(
        serialization_alias="processingSpecification"
    )
    notifications_url: Optional[str] = Field(
        serialization_alias="notificationsURL",
        default=None,
    )


class LogSeverity(str, Enum):
    INFO = "info"
    ERROR = "error"
    WARNING = "warning"


class JobLog(BaseModel):
    created_at: datetime = Field(alias="createdAt")
    processing_stage_id: Optional[str] = Field(alias="processingStageId", default=None)
    task_id: Optional[str] = Field(alias="taskId", default=None)
    severity: LogSeverity
    reporting_module: str = Field(alias="reportingModule")
    log_type: str = Field(alias="logType")
    payload: dict


class JobLogsResponse(BaseModel):
    logs: List[JobLog]
    next_page_token: Optional[str] = Field(alias="nextPageToken", default=None)
