from datetime import date, datetime
from typing import List

from pydantic import BaseModel, Field


class ShardDetails(BaseModel):
    shard_id: str = Field(alias="shardId")
    status_name: str = Field(alias="statusName")
    status_type: str = Field(alias="statusType")
    is_terminal: bool = Field(alias="isTerminal")
    event_timestamp: datetime = Field(alias="eventTimestamp")
    shard_objects_count: int = Field(alias="shardObjectsCount")


class BatchDetails(BaseModel):
    display_name: str = Field(alias="displayName")
    batch_id: str = Field(alias="batchId")
    batch_type: str = Field(alias="batchType")
    batch_content_type: str = Field(alias="batchContentType")
    expiry_date: date = Field(alias="expiryDate")


class ListBatchesResponse(BaseModel):
    batches: List[BatchDetails]
