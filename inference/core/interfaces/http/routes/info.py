from typing import Optional

from fastapi import APIRouter, HTTPException, Query

rom inference.core.version import __version__
from inference.core.devices.utils import GLOBAL_INFERENCE_SERVER_ID
from inference.core.entities.responses.server_state import ServerVersionInfo


def create_info_router() -> APIRouter:
    router = APIRouter()

    @router.get(
        "/info",
        response_model=ServerVersionInfo,
        summary="Info",
        description="Get the server name and version number",
    )
    def root():
        """Endpoint to get the server name and version number.

        Returns:
            ServerVersionInfo: The server version information.
        """
        return ServerVersionInfo(
            name="Roboflow Inference Server",
            version=__version__,
            uuid=GLOBAL_INFERENCE_SERVER_ID,
        )

    @router.get(
        "/logs",
        summary="Get Recent Logs",
        description="Get recent application logs for debugging",
    )
    def get_logs(
        limit: Optional[int] = Query(
            100, description="Maximum number of log entries to return"
        ),
        level: Optional[str] = Query(
            None,
            description="Filter by log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
        ),
        since: Optional[str] = Query(
            None, description="Return logs since this ISO timestamp"
        ),
    ):
        """Only available when ENABLE_IN_MEMORY_LOGS is set to 'true'."""
        from inference.core.logging.memory_handler import (
            get_recent_logs,
            is_memory_logging_enabled,
        )

        if not is_memory_logging_enabled():
            raise HTTPException(
                status_code=404, detail="Logs endpoint not available"
            )

        try:
            logs = get_recent_logs(limit=limit or 100, level=level, since=since)
            return {"logs": logs, "total_count": len(logs)}
        except (ImportError, ModuleNotFoundError):
            raise HTTPException(
                status_code=500, detail="Logging system not properly initialized"
            )

    return router

