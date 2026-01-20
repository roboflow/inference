"""
GCP Serverless Logging Event Definitions.

This module defines structured event dataclasses for GCP Cloud Logging.
Each event type captures specific model lifecycle information.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Literal, Optional


@dataclass
class BaseGCPEvent:
    """Base class for all GCP serverless logging events."""

    event_type: str
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class MemorySnapshot:
    """Detailed memory state snapshot (only populated when DETAILED_MEMORY=true)."""

    gpu_allocated_before: Optional[int] = None
    gpu_allocated_after: Optional[int] = None
    gpu_reserved_after: Optional[int] = None
    gpu_free: Optional[int] = None
    gpu_total: Optional[int] = None
    process_rss_bytes: Optional[int] = None
    system_available_bytes: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RequestReceivedEvent(BaseGCPEvent):
    """Event: Direct inference request received."""

    event_type: str = field(default="request_received", init=False)
    model_id: Optional[str] = None
    api_key_hash: Optional[str] = None
    endpoint_type: Optional[str] = None
    invocation_source: str = "direct"


@dataclass
class WorkflowRequestReceivedEvent(BaseGCPEvent):
    """Event: Workflow request received."""

    event_type: str = field(default="workflow_request_received", init=False)
    workflow_id: Optional[str] = None
    workflow_instance_id: Optional[str] = None
    api_key_hash: Optional[str] = None
    step_count: Optional[int] = None


@dataclass
class ModelCacheStatusEvent(BaseGCPEvent):
    """Event: Model cache status check (per request)."""

    event_type: str = field(default="model_cache_status", init=False)
    model_id: str = ""
    cache_hit: bool = False
    disk_cache_hit: Optional[bool] = None
    invocation_source: str = "direct"
    workflow_instance_id: Optional[str] = None
    workflow_id: Optional[str] = None
    step_name: Optional[str] = None


@dataclass
class ModelLoadedToDiskEvent(BaseGCPEvent):
    """Event: Model artifacts downloaded to disk."""

    event_type: str = field(default="model_loaded_to_disk", init=False)
    model_id: str = ""
    package_id: Optional[str] = None
    backend: Optional[str] = None
    download_bytes: int = 0
    download_duration_ms: float = 0.0
    artifact_count: int = 0


@dataclass
class ModelLoadedToMemoryEvent(BaseGCPEvent):
    """Event: Model loaded to memory (GPU/CPU)."""

    event_type: str = field(default="model_loaded_to_memory", init=False)
    model_id: str = ""
    backend: Optional[str] = None
    device: Optional[str] = None
    load_duration_ms: float = 0.0
    model_architecture: Optional[str] = None
    model_footprint_bytes: int = 0
    memory: Optional[MemorySnapshot] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary, handling nested memory snapshot."""
        result = {k: v for k, v in asdict(self).items() if v is not None}
        if self.memory is not None:
            memory_dict = self.memory.to_dict()
            if memory_dict:
                result["memory"] = memory_dict
            else:
                result.pop("memory", None)
        return result


@dataclass
class ModelEvictedEvent(BaseGCPEvent):
    """Event: Model evicted from memory."""

    event_type: str = field(default="model_evicted", init=False)
    model_id: str = ""
    reason: Literal["capacity", "memory_pressure", "explicit"] = "capacity"
    lifetime_seconds: float = 0.0
    inference_count: int = 0
    memory: Optional[MemorySnapshot] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary, handling nested memory snapshot."""
        result = {k: v for k, v in asdict(self).items() if v is not None}
        if self.memory is not None:
            memory_dict = self.memory.to_dict()
            if memory_dict:
                result["memory"] = memory_dict
            else:
                result.pop("memory", None)
        return result


@dataclass
class InferenceCompletedEvent(BaseGCPEvent):
    """Event: Inference completed successfully."""

    event_type: str = field(default="inference_completed", init=False)
    model_id: str = ""
    inference_duration_ms: float = 0.0
    batch_size: int = 1
    cache_hit: Optional[bool] = None  # Whether model was already loaded (from add_model)
    invocation_source: str = "direct"
    workflow_instance_id: Optional[str] = None
    workflow_id: Optional[str] = None
    step_name: Optional[str] = None
