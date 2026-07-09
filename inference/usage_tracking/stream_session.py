"""Per-pipeline stream session identity for usage tracking.

The usage collector's ``exec_session_id`` is minted once per process, so every
video pipeline running in the same inference server reports usage under one
session. Downstream stream billing counts concurrent sessions, which collapses
all of a device's cameras into a single billable stream.

``stream_session_id`` is a context variable scoped to a single pipeline's
inference thread. Anything recorded while it is set (workflow runs, model
inferences) is attributed to that pipeline, so concurrently running pipelines
stay distinguishable even when they share a process, an API key, and a
workflow.

This module must stay import-light: it is imported both by the usage collector
and by ``InferencePipeline``.
"""

import time
from contextvars import ContextVar
from typing import Optional
from uuid import uuid4

stream_session_id: ContextVar[Optional[str]] = ContextVar(
    "stream_session_id", default=None
)


def mint_stream_session_id() -> str:
    # Same shape as UsageCollector's exec_session_id so downstream consumers
    # can treat both identifiers uniformly.
    return f"{time.time_ns()}_{uuid4().hex[:4]}"
