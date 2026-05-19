"""FastAPI dependencies.

Kept separate from `app.py` to avoid circular imports — routers need
`get_model_manager`, and `app.py` imports routers.
"""

from __future__ import annotations

from fastapi import Request

from inference_server.proxies.base import ModelManagerProxy


def get_model_manager(request: Request) -> ModelManagerProxy:
    """Proxy is set on app.state in `app._lifespan`."""
    return request.app.state.model_manager
