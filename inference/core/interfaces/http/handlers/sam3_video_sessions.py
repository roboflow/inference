import asyncio
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse

from inference.core.interfaces.http.api_key_resolution import (
    api_key_fallback,
    api_key_override,
)
from inference.core.interfaces.http.error_handlers import with_route_exceptions_async
from inference.core.interfaces.sam3_video_session.entities import (
    Sam3VideoInternalEventRequest,
    Sam3VideoSessionCreated,
    Sam3VideoSessionEndRequest,
    Sam3VideoSessionRequest,
    Sam3VideoSessionSnapshot,
)
from inference.core.interfaces.sam3_video_session.service import (
    end_session,
    iter_public_events,
    publish_internal_event,
    resolve_events_callback_base,
    session_snapshot,
    start_session,
)


def _require_api_key(api_key: Optional[str]) -> str:
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "unauthorized"},
        )
    return api_key


def _map_lookup_errors(error: Exception) -> None:
    if isinstance(error, KeyError):
        raise HTTPException(
            status_code=404,
            detail={"status": "error", "message": "session not found"},
        ) from error
    if isinstance(error, PermissionError):
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "unauthorized"},
        ) from error
    if isinstance(error, ValueError):
        raise HTTPException(
            status_code=400,
            detail={"status": "error", "message": str(error)},
        ) from error
    raise error


def register_sam3_video_session_routes(app: FastAPI) -> None:
    @app.post(
        "/sam3/video/sessions",
        response_model=Sam3VideoSessionCreated,
        summary="Start a SAM3 video tracking session",
        description=(
            "Spawns a long-lived Modal (or local) SAM3 worker that downloads "
            "video_url, writes track artifacts to GCS via Roboflow-signed "
            "upload URLs, and publishes overlay events. Auth: Roboflow API key "
            "via query, Authorization Bearer header, or JSON body. Reachable "
            "on serverless."
        ),
    )
    @with_route_exceptions_async
    async def create_sam3_video_session(
        request: Sam3VideoSessionRequest,
    ) -> Sam3VideoSessionCreated:
        api_key = _require_api_key(api_key_override(request.api_key))
        try:
            callback_base = resolve_events_callback_base(request.events_callback_base)
        except ValueError as error:
            _map_lookup_errors(error)
        session_id = await asyncio.to_thread(
            start_session,
            request,
            api_key=api_key,
            events_callback_base=callback_base,
        )
        return Sam3VideoSessionCreated(session_id=session_id)

    @app.get(
        "/sam3/video/sessions/{session_id}",
        response_model=Sam3VideoSessionSnapshot,
        summary="SAM3 video session snapshot",
        description="Auth: Roboflow API key of the session owner. Reachable on serverless.",
    )
    @with_route_exceptions_async
    async def get_sam3_video_session(
        session_id: str,
        api_key: Optional[str] = Query(None),
    ) -> Sam3VideoSessionSnapshot:
        resolved_key = _require_api_key(api_key_fallback(api_key))
        try:
            snap = await asyncio.to_thread(
                session_snapshot,
                session_id,
                api_key=resolved_key,
            )
        except (KeyError, PermissionError) as error:
            _map_lookup_errors(error)
        if snap is None:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "session not found"},
            )
        return Sam3VideoSessionSnapshot.model_validate(snap)

    @app.get(
        "/sam3/video/sessions/{session_id}/events",
        summary="SAM3 video session event stream",
        description=(
            "Server-sent events for live overlay. Auth: Roboflow API key of "
            "the session owner via query or Authorization Bearer header "
            "(EventSource cannot set headers; prefer fetch). Reachable on "
            "serverless."
        ),
    )
    @with_route_exceptions_async
    async def stream_sam3_video_session_events(
        session_id: str,
        api_key: Optional[str] = Query(None),
        after: int = Query(0, ge=0),
    ) -> StreamingResponse:
        resolved_key = _require_api_key(api_key_fallback(api_key))
        try:
            await asyncio.to_thread(
                session_snapshot,
                session_id,
                api_key=resolved_key,
            )
        except (KeyError, PermissionError) as error:
            _map_lookup_errors(error)
        return StreamingResponse(
            iter_public_events(
                session_id,
                api_key=resolved_key,
                after_seq=after,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    @app.post(
        "/sam3/video/sessions/{session_id}/end",
        summary="End a SAM3 video tracking session",
        description="Auth: Roboflow API key of the session owner. Reachable on serverless.",
    )
    @with_route_exceptions_async
    async def end_sam3_video_session(
        session_id: str,
        request: Sam3VideoSessionEndRequest,
    ) -> dict:
        api_key = _require_api_key(api_key_override(request.api_key))
        try:
            await asyncio.to_thread(end_session, session_id, api_key=api_key)
        except (KeyError, PermissionError) as error:
            _map_lookup_errors(error)
        return {"status": "ok"}

    @app.post(
        "/sam3/video/sessions/{session_id}/internal/events",
        summary="SAM3 worker event publish",
        description=(
            "Called by the SAM3 video worker. Auth: Roboflow API key plus the "
            "session publish_token. Reachable on serverless."
        ),
    )
    @with_route_exceptions_async
    async def publish_sam3_video_internal_event(
        session_id: str,
        request: Sam3VideoInternalEventRequest,
        api_key: Optional[str] = Query(None),
    ) -> dict:
        _require_api_key(api_key_override(api_key))
        try:
            stop_requested = await asyncio.to_thread(
                publish_internal_event,
                session_id,
                publish_token=request.publish_token,
                event=request.event,
            )
        except (KeyError, PermissionError) as error:
            _map_lookup_errors(error)
        return {"stop_requested": stop_requested}
