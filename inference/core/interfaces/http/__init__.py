from contextvars import ContextVar
from typing import Optional

# Set by HTTP middleware so downstream code (e.g. model manager) can
# identify which request path triggered a model load.
current_request_path: ContextVar[Optional[str]] = ContextVar(
    "current_request_path", default=None
)
