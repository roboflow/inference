from datetime import date, datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class ShardDetails(BaseModel):
    shard_id: str = Field(alias="shardId")
    status_name: str = Field(alias="statusName")
    status_type: str = Field(alias="statusType")
    is_terminal: bool = Field(alias="isTerminal")
    event_timestamp: datetime = Field(alias="eventTimestamp")
    shard_objects_count: int = Field(alias="shardObjectsCount")


class BatchMetadata(BaseModel):
    display_name: str = Field(alias="displayName")
    batch_id: str = Field(alias="batchId")
    batch_type: str = Field(alias="batchType")
    batch_content_type: str = Field(alias="batchContentType")
    created_date: date = Field(alias="createdDate")
    expiry_date: date = Field(alias="expiryDate")


class ListBatchesResponse(BaseModel):
    batches: List[BatchMetadata]
    next_page_token: Optional[str] = Field(alias="nextPageToken", default=None)


class MultipartBatchPartMetadata(BaseModel):
    part_name: str = Field(alias="partName")
    part_type: str = Field(alias="partType")
    content_type: str = Field(alias="contentType")


class ListMultipartBatchPartsResponse(BaseModel):
    batch_parts: List[MultipartBatchPartMetadata] = Field(alias="batchParts")


class BatchExportResponse(BaseModel):
    urls: List[str]
    next_page_token: Optional[str] = Field(alias="nextPageToken", default=None)
