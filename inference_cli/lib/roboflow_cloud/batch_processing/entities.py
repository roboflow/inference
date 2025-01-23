from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class JobMetadata(BaseModel):
    id: str
    name: str
    job_definition: dict = Field(alias="jobDefinition")
    current_stage: Optional[str] = Field(alias="currentStage", default=None)
    planned_stages: Optional[List[str]] = Field(alias="plannedStages", default=None)
    error: bool = Field(default=False)
    is_terminal: bool = Field(alias="isTerminal")
    last_notification: str = Field(alias="lastNotification")
    created_at: datetime = Field(alias="createdAt")
    last_update: datetime = Field(alias="lastUpdate")


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
    status_name: str = Field(alias="statusName")
    status_type: str = Field(alias="statusType")
    is_terminal: bool = Field(alias="isTerminal", default=False)
    last_event_timestamp: datetime = Field(alias="lastEventTimestamp")


class ListJobStagesResponse(BaseModel):
    stages: List[JobStageDetails]


class TaskStatus(BaseModel):
    task_id: str = Field(alias="taskId")
    status_name: str = Field(alias="statusName")
    status_type: str = Field(alias="statusType")
    is_terminal: bool = Field(alias="isTerminal")
    event_timestamp: datetime = Field(alias="eventTimestamp")


class ListJobStageTasksResponse(BaseModel):
    tasks: List[TaskStatus]
    next_page_token: Optional[str] = Field(alias="nextToken")


class MultipartBatchPartMetadata(BaseModel):
    part_name: str = Field(alias="partName")
    part_type: str = Field(alias="partType")
    content_type: str = Field(alias="contentType")


class ListMultipartBatchPartsResponse(BaseModel):
    batch_parts: List[MultipartBatchPartMetadata] = Field(alias="batchParts")
