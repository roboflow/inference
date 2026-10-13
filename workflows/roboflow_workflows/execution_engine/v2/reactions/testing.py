"""Deprecated import path for ``roboflow_workflows.v2.engine.reactions.testing``.

Kept so existing V2 alpha callers keep working; it re-exports the same
objects. Import from the canonical module instead.
"""

from roboflow_workflows.v2.engine.reactions.testing import (
    CapturedCall,
    CapturedEvent,
    block_call,
)

__all__ = [
    "CapturedCall",
    "CapturedEvent",
    "block_call",
]
