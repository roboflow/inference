from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class DataSource(str, Enum):
    LOCAL_DIRECTORY = "local-directory"
    REFERENCES_FILE = "references-file"


class ShardDetails(BaseModel):
    shard_id: str = Field(alias="shardId")
    status_name: str = Field(alias="statusName")
    status_type: str = Field(alias="statusType")
    event_timestamp: datetime = Field(alias="eventTimestamp")
    shard_objects_count: int = Field(alias="shardObjectsCount")
    status_metadata: Optional[dict] = Field(alias="statusMetadata", default=None)


class BatchMetadata(BaseModel):
    display_name: str = Field(alias="displayName")
    batch_id: str = Field(alias="batchId")
    batch_type: str = Field(alias="batchType")
    batch_content_type: str = Field(alias="batchContentType")
    created_date: datetime = Field(alias="createdDate")
    expiry_date: datetime = Field(alias="expiryDate")


class ListBatchesResponse(BaseModel):
    batches: List[BatchMetadata]
    next_page_token: Optional[str] = Field(alias="nextPageToken", default=None)


class MultipartBatchPartMetadata(BaseModel):
    part_name: str = Field(alias="partName")
    part_type: str = Field(alias="partType")
    content_type: str = Field(alias="contentType")
    part_description: Optional[str] = Field(alias="partDescription", default=None)
    nestedContentType: Optional[str] = Field(alias="nestedContentType", default=None)


class ListMultipartBatchPartsResponse(BaseModel):
    batch_parts: List[MultipartBatchPartMetadata] = Field(alias="batchParts")


class BatchExportResponse(BaseModel):
    urls: List[str]
    next_page_token: Optional[str] = Field(alias="nextPageToken", default=None)


class FileMetadata(BaseModel):
    download_url: str = Field(alias="downloadURL", serialization_alias="downloadURL")
    file_name: str = Field(alias="fileName", serialization_alias="fileName")
    part_name: Optional[str] = Field(
        default=None, alias="partName", serialization_alias="partName"
    )
    shard_id: Optional[str] = Field(
        default=None, alias="shardId", serialization_alias="shardId"
    )
    content_type: str = Field(alias="contentType", serialization_alias="contentType")
    nested_content_type: Optional[str] = Field(
        default=None, alias="nestedContentType", serialization_alias="nestedContentType"
    )


class ListBatchResponse(BaseModel):
    files_metadata: List[FileMetadata] = Field(alias="filesMetadata")
    next_page_token: Optional[str] = Field(alias="nextPageToken", default=None)


class DownloadLogEntry(BaseModel):
    file_metadata: FileMetadata
    local_path: str


class ImageReferencesIngestResponse(BaseModel):
    shard_ids: Optional[List[str]] = Field(alias="shardIds", default=None)
    ingest_id: str = Field(alias="ingestId")
    duplicated: Optional[bool] = Field(default=None)


class VideoReferencesIngestResponse(BaseModel):
    ingest_id: str = Field(alias="ingestId")
    duplicated: Optional[bool] = Field(default=None)


class PageOfBatchShardsStatuses(BaseModel):
    shards: List[ShardDetails]
    next_page_token: Optional[str] = Field(alias="nextPageToken", default=None)
