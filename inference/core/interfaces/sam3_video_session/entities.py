from typing import Any, Dict, List, Literal, Optional, Union
from urllib.parse import urlparse

from pydantic import BaseModel, Field, field_validator

SAM3_VIDEO_EVENT_DOWNLOADING = "downloading"
SAM3_VIDEO_EVENT_FRAME = "frame"
SAM3_VIDEO_EVENT_CHECKPOINTED = "checkpointed"
SAM3_VIDEO_EVENT_DONE = "done"
SAM3_VIDEO_EVENT_ERROR = "error"

Sam3VideoEventType = Literal[
    "downloading",
    "frame",
    "checkpointed",
    "done",
    "error",
]

Sam3VideoSessionStatus = Literal[
    "starting",
    "downloading",
    "running",
    "completed",
    "failed",
    "cancelled",
]

CHUNK_SAMPLE_SIZE = 500
DEFAULT_THRESHOLD = 0.35
DEFAULT_CLASS_NAME = "object"
SESSION_EVENT_TTL_SECONDS = 600
MAX_VIDEO_BYTES = 2 * 1024 * 1024 * 1024
MODEL_ID = "sam3video"

ALLOWED_APP_HOSTS = {
    "app.roboflow.com",
    "app.roboflow.one",
    "localhost",
    "127.0.0.1",
}
ALLOWED_APP_HOST_SUFFIXES = (".roboflow.com", ".roboflow.one")
ARTIFACT_CHUNK_CONTENT_TYPE = "application/json"


def is_allowed_app_base_url(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False
    host = (parsed.hostname or "").lower()
    if not host:
        return False
    if host in ALLOWED_APP_HOSTS:
        return True
    return host.endswith(ALLOWED_APP_HOST_SUFFIXES)


def validated_app_base_url(url: str) -> str:
    if not is_allowed_app_base_url(url):
        raise ValueError("app_base_url must be a Roboflow app host")
    return url.rstrip("/") + "/"


class Sam3VideoTimeBase(BaseModel):
    numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)


class Sam3VideoArtifactTarget(BaseModel):
    app_base_url: str
    video_id: str
    workspace_id: str
    dataset_id: str
    revision_id: str
    video_time_base: Optional[Sam3VideoTimeBase] = None

    @field_validator("app_base_url")
    @classmethod
    def validate_app_base_url_field(cls, value: str) -> str:
        return validated_app_base_url(value).rstrip("/")


class Sam3VideoSessionRequest(BaseModel):
    video_url: str
    class_names: List[str]
    artifact: Sam3VideoArtifactTarget
    api_key: Optional[str] = None
    requested_plan: Optional[str] = None
    requested_region: Optional[str] = None
    processing_timeout: Optional[int] = None
    threshold: float = DEFAULT_THRESHOLD
    events_callback_base: Optional[str] = None


class Sam3VideoSessionCreated(BaseModel):
    session_id: str


class Sam3VideoSessionSnapshot(BaseModel):
    session_id: str
    status: Sam3VideoSessionStatus
    last_seq: int = 0
    last_frame_id: Optional[int] = None
    error_message: Optional[str] = None
    stop_requested: bool = False


class Sam3VideoInternalEventRequest(BaseModel):
    publish_token: str
    event: Dict[str, Any]


class Sam3VideoInternalEventResponse(BaseModel):
    stop_requested: bool = False


class Sam3VideoSessionEndRequest(BaseModel):
    api_key: Optional[str] = None


class Sam3VideoWorkerPayload(BaseModel):
    session_id: str
    video_url: str
    class_names: List[str]
    artifact: Sam3VideoArtifactTarget
    api_key: Optional[str] = None
    threshold: float = DEFAULT_THRESHOLD
    events_callback_url: str
    publish_token: str
    requested_plan: Optional[str] = None
    workspace_id: Optional[str] = None
    processing_timeout: Optional[int] = None


JsonDict = Dict[str, Any]
JsonList = List[Union[JsonDict, Any]]
