from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class BatchJobStateDescription(BaseModel):
    job_id: str = Field(alias="jobId")
    display_name: str = Field(alias="displayName")
    stage_name: Optional[str] = Field(alias="stageName", default=None)
    planned_stages: Optional[List[str]] = Field(alias="plannedStages", default=None)
    error: bool = Field(default=False)
    error_details: Optional[str] = Field(alias="errorDetails", default=None)


class ListBatchJobsResponse(BaseModel):
    batch_jobs: List[BatchJobStateDescription] = Field(alias="batchJobs")


class BatchJobMetadata(BaseModel):
    job_type: str = Field(alias="jobType")
    display_name: str = Field(alias="displayName")
    job_parameters: dict = Field(alias="jobParameters")
    input_definition: dict = Field(alias="inputDefinition")
    stage_name: Optional[str] = Field(alias="stageName", default=None)
    event_timestamp: datetime = Field(alias="eventTimestamp")
    planned_stages: Optional[List[str]] = Field(alias="plannedStages", default=None)


class BatchJobMetadataResponse(BaseModel):
    job_metadata: BatchJobMetadata = Field(alias="jobMetadata")


class JobStageDetails(BaseModel):
    processing_stage_id: str = Field(alias="processingStageId")
    processing_stage_name: str = Field(alias="processingStageName")
    tasks_number: int = Field(alias="tasksNumber")
    output_batches: List[str] = Field(alias="outputBatches")
    start_timestamp: datetime = Field(alias="startTimestamp")
    status_name: str = Field(alias="statusName")
    status_type: str = Field(alias="statusType")
    is_terminal: bool = Field(alias="isTerminal", default=False)
    last_event_timestamp: datetime = Field(alias="lastEventTimestamp")


class ListJobStagesResponse(BaseModel):
    job_stages_metadata: List[JobStageDetails] = Field(alias="jobStagesMetadata")


class TaskStatus(BaseModel):
    task_id: str = Field(alias="taskId")
    status_name: str = Field(alias="statusName")
    status_type: str = Field(alias="statusType")
    is_terminal: bool = Field(alias="isTerminal")
    event_timestamp: datetime = Field(alias="eventTimestamp")


class ListJobStageTasksResponse(BaseModel):
    total_tasks_number: int = Field(alias="totalTasksNumber")
    tasks_statuses: List[TaskStatus] = Field(alias="tasksStatuses")


class MultipartBatchPartMetadata(BaseModel):
    part_name: str = Field(alias="partName")
    part_type: str = Field(alias="partType")
    content_type: str = Field(alias="contentType")


class ListMultipartBatchPartsResponse(BaseModel):
    batch_parts: List[MultipartBatchPartMetadata] = Field(alias="batchParts")
