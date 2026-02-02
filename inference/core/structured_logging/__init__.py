"""
Structured Event Logging Module.

This module provides structured logging for serverless deployments.
It captures model lifecycle events for observability and analysis.

Usage:
    from inference.core.structured_logging import structured_logger

    # The logger is enabled when STRUCTURED_LOGGING_ENABLED=True

Environment Variables:
    STRUCTURED_LOGGING_ENABLED: Master switch (default: False)
    STRUCTURED_LOGGING_SAMPLE_RATE: Sample rate for high-volume events (default: 1.0)
    STRUCTURED_LOGGING_DETAILED_MEMORY: Enable detailed memory introspection (default: False)
"""

from inference.core.structured_logging.context import (
    RequestContext,
    clear_request_context,
    get_request_context,
    hash_api_key,
    set_request_context,
    update_request_context,
)
from inference.core.structured_logging.events import (
    BaseEvent,
    InferenceCompletedEvent,
    MemorySnapshot,
    ModelCacheStatusEvent,
    ModelEvictedEvent,
    ModelLoadedToDiskEvent,
    ModelLoadedToMemoryEvent,
    RequestReceivedEvent,
    WorkflowRequestReceivedEvent,
)
from inference.core.structured_logging.logger import StructuredLogger, structured_logger
from inference.core.structured_logging.memory import (
    get_gpu_allocated,
    measure_memory_after_load,
    measure_memory_before_load,
    measure_memory_for_eviction,
)
from inference.core.structured_logging.access_manager import LoggingModelAccessManager

__all__ = [
    # Logger
    "structured_logger",
    "StructuredLogger",
    # Events
    "BaseEvent",
    "RequestReceivedEvent",
    "WorkflowRequestReceivedEvent",
    "ModelCacheStatusEvent",
    "ModelLoadedToDiskEvent",
    "ModelLoadedToMemoryEvent",
    "ModelEvictedEvent",
    "InferenceCompletedEvent",
    "MemorySnapshot",
    # Context
    "RequestContext",
    "set_request_context",
    "get_request_context",
    "clear_request_context",
    "update_request_context",
    "hash_api_key",
    # Memory
    "get_gpu_allocated",
    "measure_memory_before_load",
    "measure_memory_after_load",
    "measure_memory_for_eviction",
    # Access Manager (for inference-models integration)
    "LoggingModelAccessManager",
]
