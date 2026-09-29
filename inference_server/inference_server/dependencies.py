"""FastAPI dependencies.

Kept separate from `app.py` to avoid circular imports — routers need
`get_model_manager`, and `app.py` imports routers.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request


def get_model_manager(request: Request) -> Any:
    """Proxy is set on app.state in `app._lifespan`."""
    return request.app.state.model_manager
