"""
GCP Serverless Structured Logging Module.

This module provides structured logging for GCP serverless deployments.
It captures model lifecycle events for observability and analysis.

Usage:
    from inference.core.gcp_logging import gcp_logger, log_model_loaded_to_memory

    # The logger is automatically enabled when GCP_SERVERLESS=True
    # and GCP_LOGGING_ENABLED=True (default when GCP_SERVERLESS)

Environment Variables:
    GCP_LOGGING_ENABLED: Master switch (default: True when GCP_SERVERLESS)
    GCP_LOGGING_SAMPLE_RATE: Sample rate for high-volume events (default: 1.0)
    GCP_LOGGING_DETAILED_MEMORY: Enable detailed memory introspection (default: False)
    GCP_LOGGING_DEBUG: Output logs to stderr instead of stdout for local visibility (default: False)
"""

from inference.core.gcp_logging.context import (
    GCPRequestContext,
    clear_gcp_context,
    get_gcp_context,
    hash_api_key,
    set_gcp_context,
    update_gcp_context,
)
from inference.core.gcp_logging.events import (
    BaseGCPEvent,
    InferenceCompletedEvent,
    MemorySnapshot,
    ModelCacheStatusEvent,
    ModelEvictedEvent,
    ModelLoadedToDiskEvent,
    ModelLoadedToMemoryEvent,
    RequestReceivedEvent,
    WorkflowRequestReceivedEvent,
)
from inference.core.gcp_logging.logger import GCPServerlessLogger, gcp_logger
from inference.core.gcp_logging.memory import (
    get_gpu_allocated,
    measure_memory_after_load,
    measure_memory_before_load,
    measure_memory_for_eviction,
)
from inference.core.gcp_logging.access_manager import GCPLoggingModelAccessManager

__all__ = [
    # Logger
    "gcp_logger",
    "GCPServerlessLogger",
    # Events
    "BaseGCPEvent",
    "RequestReceivedEvent",
    "WorkflowRequestReceivedEvent",
    "ModelCacheStatusEvent",
    "ModelLoadedToDiskEvent",
    "ModelLoadedToMemoryEvent",
    "ModelEvictedEvent",
    "InferenceCompletedEvent",
    "MemorySnapshot",
    # Context
    "GCPRequestContext",
    "set_gcp_context",
    "get_gcp_context",
    "clear_gcp_context",
    "update_gcp_context",
    "hash_api_key",
    # Memory
    "get_gpu_allocated",
    "measure_memory_before_load",
    "measure_memory_after_load",
    "measure_memory_for_eviction",
    # Access Manager (for inference-models integration)
    "GCPLoggingModelAccessManager",
]
