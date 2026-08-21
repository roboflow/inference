import hashlib
import multiprocessing
import secrets
import uuid
from typing import Any, Dict, Optional
from urllib.parse import urljoin

from inference.core.env import (
    WEBRTC_MODAL_TOKEN_ID,
    WEBRTC_MODAL_TOKEN_SECRET,
    WEBRTC_MODAL_USAGE_QUOTA_ENABLED,
    WEBRTC_WORKSPACE_STREAM_QUOTA,
    WEBRTC_WORKSPACE_STREAM_QUOTA_ENABLED,
    WEBRTC_WORKSPACE_STREAM_TTL_SECONDS,
)
from inference.core.exceptions import CreditsExceededError, WorkspaceStreamQuotaError
from inference.core.interfaces.sam3_video_session.entities import (
    Sam3VideoSessionRequest,
    Sam3VideoWorkerPayload,
)
from inference.core.interfaces.sam3_video_session.session_store import (
    append_event,
    create_session,
    get_session,
    is_stop_requested,
    mark_client_seen,
    request_stop,
    snapshot,
    update_session,
)
from inference.core.interfaces.sam3_video_session.worker import (
    run_sam3_video_session_from_dict,
)
from inference.core.interfaces.webrtc_worker.utils import (
    is_over_quota,
    is_over_workspace_session_quota,
    register_webrtc_session,
)
from inference.core.logger import logger
from inference.core.roboflow_api import get_roboflow_workspace


def _hash_api_key(api_key: Optional[str]) -> str:
    if not api_key:
        return ""
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def _events_callback_url(base: str, session_id: str) -> str:
    return urljoin(
        base.rstrip("/") + "/", f"sam3/video/sessions/{session_id}/internal/events"
    )


def _local_worker_process(payload: Dict[str, Any]) -> None:
    run_sam3_video_session_from_dict(payload)


def _spawn_local(payload: Sam3VideoWorkerPayload) -> None:
    ctx = multiprocessing.get_context("spawn")
    process = ctx.Process(
        target=_local_worker_process,
        args=(payload.model_dump(mode="json"),),
        daemon=False,
    )
    process.start()


def _spawn_modal(payload: Sam3VideoWorkerPayload) -> Optional[str]:
    from inference.core.interfaces.webrtc_worker.modal import (
        spawn_sam3_video_session_modal,
    )

    return spawn_sam3_video_session_modal(payload)


def start_session(
    request: Sam3VideoSessionRequest,
    *,
    api_key: Optional[str],
    events_callback_base: str,
) -> str:
    if WEBRTC_MODAL_USAGE_QUOTA_ENABLED and api_key and is_over_quota(api_key):
        raise CreditsExceededError("API key over quota")

    workspace_id = None
    if api_key:
        workspace_id = get_roboflow_workspace(api_key=api_key)

    session_id = str(uuid.uuid4())
    if WEBRTC_WORKSPACE_STREAM_QUOTA_ENABLED and workspace_id:
        if is_over_workspace_session_quota(
            workspace_id=workspace_id,
            quota=WEBRTC_WORKSPACE_STREAM_QUOTA,
            ttl_seconds=WEBRTC_WORKSPACE_STREAM_TTL_SECONDS,
        ):
            raise WorkspaceStreamQuotaError(
                f"You have reached the maximum of {WEBRTC_WORKSPACE_STREAM_QUOTA} "
                f"concurrent streams."
            )
        register_webrtc_session(workspace_id=workspace_id, session_id=session_id)

    publish_token = secrets.token_urlsafe(32)
    create_session(
        session_id,
        workspace_id=workspace_id,
        publish_token=publish_token,
        owner_api_key_hash=_hash_api_key(api_key),
    )

    callback_base = request.events_callback_base or events_callback_base
    payload = Sam3VideoWorkerPayload(
        session_id=session_id,
        video_url=request.video_url,
        class_names=request.class_names,
        artifact=request.artifact,
        api_key=api_key,
        threshold=request.threshold,
        events_callback_url=_events_callback_url(callback_base, session_id),
        publish_token=publish_token,
        requested_plan=request.requested_plan,
        workspace_id=workspace_id,
        processing_timeout=request.processing_timeout,
    )

    if WEBRTC_MODAL_TOKEN_ID and WEBRTC_MODAL_TOKEN_SECRET:
        try:
            call_id = _spawn_modal(payload)
            if call_id:
                update_session(session_id, modal_call_id=call_id)
        except Exception:
            request_stop(session_id)
            raise
    else:
        _spawn_local(payload)

    logger.info("Started SAM3 video session %s", session_id)
    return session_id


def publish_internal_event(
    session_id: str,
    *,
    publish_token: str,
    event: Dict[str, Any],
) -> bool:
    meta = get_session(session_id)
    if meta is None:
        raise KeyError(f"Unknown SAM3 video session {session_id}")
    stored = str(meta.get("publish_token") or "")
    if len(stored) != len(publish_token) or not secrets.compare_digest(
        stored, publish_token
    ):
        raise PermissionError("Invalid SAM3 video session publish token")
    event_type = str(event.get("type") or "")
    payload = {key: value for key, value in event.items() if key != "type"}
    append_event(session_id, event_type, payload or None)
    return is_stop_requested(session_id, WEBRTC_WORKSPACE_STREAM_TTL_SECONDS)


def end_session(session_id: str) -> None:
    meta = request_stop(session_id)
    call_id = meta.get("modal_call_id") if meta else None
    if call_id:
        try:
            import modal

            modal.FunctionCall.from_id(call_id).cancel()
        except Exception as error:
            logger.warning("Failed to cancel SAM3 Modal call %s: %s", call_id, error)


def session_snapshot(session_id: str) -> Optional[Dict[str, Any]]:
    mark_client_seen(session_id)
    return snapshot(session_id)


def client_seen(session_id: str) -> None:
    mark_client_seen(session_id)
