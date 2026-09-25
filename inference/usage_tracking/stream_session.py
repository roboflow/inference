"""Historical home of the per-pipeline stream session identity.

The implementation lives in ``inference.core.interfaces.stream.session``; this
module re-exports the exact same objects, so the usage collector and the
pipeline keep sharing one context variable whichever name they import.
"""

from inference.core.interfaces.stream.session import (
    mint_stream_session_id,
    stream_session_id,
)

__all__ = ["mint_stream_session_id", "stream_session_id"]
