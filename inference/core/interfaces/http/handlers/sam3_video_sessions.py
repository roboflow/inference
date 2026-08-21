import asyncio
import json
from typing import AsyncIterator, Optional

from fastapi import FastAPI, HTTPException, Query, Request
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
    client_seen,
    end_session,
    publish_internal_event,
    session_snapshot,
    start_session,
)
from inference.core.interfaces.sam3_video_session.session_store import list_events
from inference.core.interfaces.webrtc_worker.utils import (
    deregister_webrtc_session,
    refresh_webrtc_session,
)
from inference.core.logger import logger


def _require_api_key(api_key: Optional[str]) -> str:
    if api_key is None:
        raise HTTPException(
            status_code=401,
            detail={"status": "error", "message": "unauthorized"},
        )
    return api_key


def _public_event(event: dict) -> dict:
    return {key: value for key, value in event.items() if key != "publish_token"}


async def _sse_stream(session_id: str, after_seq: int, workspace_id: Optional[str]) -> AsyncIterator[str]:
    last_seq = after_seq
    while True:
        client_seen(session_id)
        if workspace_id:
            refresh_webrtc_session(workspace_id=workspace_id, session_id=session_id)
        snapshot = session_snapshot(session_id)
        if snapshot is None:
            error = {"seq": last_seq, "type": "error", "message": "session not found"}
            yield f"event: error\ndata: {json.dumps(error)}\n\n"
            return
        events = list_events(session_id, after_seq=last_seq)
        for event in events:
            last_seq = int(event.get("seq") or last_seq)
            public = _public_event(event)
            yield f"event: {public.get('type', 'message')}\ndata: {json.dumps(public)}\n\n"
            if public.get("type") in {"done", "error"}:
                return
        if snapshot.get("status") in {"completed", "failed", "cancelled"}:
            return
        await asyncio.sleep(0.2)


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
        r: Request,
    ) -> Sam3VideoSessionCreated:
        api_key = api_key_override(request.api_key)
        _require_api_key(api_key)
        callback_base = str(request.events_callback_base or r.base_url)
        session_id = start_session(
            request,
            api_key=api_key,
            events_callback_base=callback_base,
        )
        return Sam3VideoSessionCreated(session_id=session_id)

    @app.get(
        "/sam3/video/sessions/{session_id}",
        response_model=Sam3VideoSessionSnapshot,
        summary="SAM3 video session snapshot",
        description="Auth: Roboflow API key. Reachable on serverless.",
    )
    @with_route_exceptions_async
    async def get_sam3_video_session(
        session_id: str,
        api_key: Optional[str] = Query(None),
    ) -> Sam3VideoSessionSnapshot:
        _require_api_key(api_key_fallback(api_key))
        snap = session_snapshot(session_id)
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
            "Server-sent events for live overlay. Auth: Roboflow API key via "
            "query or Authorization Bearer header (EventSource cannot set "
            "headers; prefer fetch). Reachable on serverless."
        ),
    )
    @with_route_exceptions_async
    async def stream_sam3_video_session_events(
        session_id: str,
        api_key: Optional[str] = Query(None),
        after: int = Query(0, ge=0),
    ) -> StreamingResponse:
        resolved_key = _require_api_key(api_key_fallback(api_key))
        snap = session_snapshot(session_id)
        if snap is None:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "session not found"},
            )
        workspace_id = None
        try:
            from inference.core.roboflow_api import get_roboflow_workspace

            workspace_id = get_roboflow_workspace(api_key=resolved_key)
        except Exception:
            logger.debug("Could not resolve workspace for SAM3 SSE heartbeat")
        return StreamingResponse(
            _sse_stream(session_id, after, workspace_id),
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
        description="Auth: Roboflow API key. Reachable on serverless.",
    )
    @with_route_exceptions_async
    async def end_sam3_video_session(
        session_id: str,
        request: Sam3VideoSessionEndRequest,
    ) -> dict:
        api_key = _require_api_key(api_key_override(request.api_key))
        end_session(session_id)
        try:
            from inference.core.roboflow_api import get_roboflow_workspace

            workspace_id = get_roboflow_workspace(api_key=api_key)
            if workspace_id:
                deregister_webrtc_session(
                    workspace_id=workspace_id,
                    session_id=session_id,
                )
        except Exception:
            logger.debug("Could not deregister SAM3 session quota slot")
        return {"status": "ok"}

    @app.post(
        "/sam3/video/sessions/{session_id}/internal/events",
        summary="Worker event publish",
        description=(
            "Called by the SAM3 worker. Auth: Roboflow API key plus the "
            "session publish_token. Reachable on serverless."
        ),
    )
    @with_route_exceptions_async
    async def publish_sam3_video_session_event(
        session_id: str,
        request: Sam3VideoInternalEventRequest,
        api_key: Optional[str] = Query(None),
    ) -> dict:
        _require_api_key(api_key_override(api_key))
        try:
            stop_requested = publish_internal_event(
                session_id,
                publish_token=request.publish_token,
                event=request.event,
            )
        except KeyError:
            raise HTTPException(
                status_code=404,
                detail={"status": "error", "message": "session not found"},
            )
        except PermissionError:
            raise HTTPException(
                status_code=401,
                detail={"status": "error", "message": "unauthorized"},
            )
        return {"stop_requested": stop_requested}
