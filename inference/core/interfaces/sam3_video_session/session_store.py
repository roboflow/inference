import json
import time
from typing import Any, Dict, List, Optional

from inference.core.cache import cache
from inference.core.env import (
    SAM3_VIDEO_SESSION_EVENT_PAGE_SIZE,
    SAM3_VIDEO_SESSION_MAX_RETAINED_EVENTS,
    WEBRTC_WORKSPACE_STREAM_TTL_SECONDS,
)
from inference.core.interfaces.sam3_video_session.entities import (
    SESSION_EVENT_TTL_SECONDS,
    Sam3VideoSessionStatus,
)

META_KEY_PREFIX = "sam3_video_session:meta:"
EVENTS_KEY_PREFIX = "sam3_video_session:events:"


def _meta_key(session_id: str) -> str:
    return f"{META_KEY_PREFIX}{session_id}"


def _events_key(session_id: str) -> str:
    return f"{EVENTS_KEY_PREFIX}{session_id}"


def _event_ttl_seconds() -> int:
    return max(SESSION_EVENT_TTL_SECONDS, WEBRTC_WORKSPACE_STREAM_TTL_SECONDS)


def _decode_meta(raw: Any) -> Optional[Dict[str, Any]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def create_session(
    session_id: str,
    *,
    workspace_id: Optional[str],
    publish_token: str,
    owner_api_key_hash: str,
) -> Dict[str, Any]:
    now = time.time()
    meta = {
        "session_id": session_id,
        "status": "starting",
        "workspace_id": workspace_id,
        "publish_token": publish_token,
        "owner_api_key_hash": owner_api_key_hash,
        "last_seq": 0,
        "last_frame_id": None,
        "error_message": None,
        "stop_requested": False,
        "last_client_seen_at": now,
        "modal_call_id": None,
        "created_at": now,
    }
    cache.set(_meta_key(session_id), meta, expire=_event_ttl_seconds())
    return meta


def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    return _decode_meta(cache.get(_meta_key(session_id)))


def _touch_meta(meta: Dict[str, Any]) -> None:
    cache.set(_meta_key(meta["session_id"]), meta, expire=_event_ttl_seconds())


def update_session(session_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
    meta = get_session(session_id)
    if meta is None:
        return None
    meta.update(fields)
    _touch_meta(meta)
    return meta


def mark_client_seen(session_id: str) -> Optional[Dict[str, Any]]:
    return update_session(session_id, last_client_seen_at=time.time())


def request_stop(session_id: str) -> Optional[Dict[str, Any]]:
    return update_session(session_id, stop_requested=True)


def is_stop_requested(session_id: str, client_ttl_seconds: int) -> bool:
    meta = get_session(session_id)
    if meta is None:
        return True
    if meta.get("stop_requested"):
        return True
    last_seen = float(meta.get("last_client_seen_at") or 0)
    return (time.time() - last_seen) > client_ttl_seconds


def append_event(
    session_id: str, event_type: str, payload: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    meta = get_session(session_id)
    if meta is None:
        raise KeyError(f"Unknown SAM3 video session {session_id}")
    seq = int(meta.get("last_seq") or 0) + 1
    event: Dict[str, Any] = {"seq": seq, "type": event_type}
    if payload:
        event.update(payload)
    events_key = _events_key(session_id)
    cache.zadd(
        events_key,
        event,
        float(seq),
        expire=_event_ttl_seconds(),
    )
    max_retained = max(1, SAM3_VIDEO_SESSION_MAX_RETAINED_EVENTS)
    if seq > max_retained:
        cache.zremrangebyscore(
            events_key,
            min=-1,
            max=float(seq - max_retained),
        )
    fields: Dict[str, Any] = {"last_seq": seq}
    if event_type == "frame":
        frame_id = event.get("frame_id")
        if frame_id is not None:
            fields["last_frame_id"] = frame_id
        fields["status"] = "running"
    elif event_type == "downloading":
        fields["status"] = "downloading"
    elif event_type == "done":
        fields["status"] = "cancelled" if event.get("cancelled") else "completed"
    elif event_type == "error":
        fields["status"] = "failed"
        fields["error_message"] = event.get("message")
    meta.update(fields)
    _touch_meta(meta)
    return event


def list_events(
    session_id: str,
    after_seq: int = 0,
    *,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    page_size = SAM3_VIDEO_SESSION_EVENT_PAGE_SIZE if limit is None else limit
    page_size = max(1, int(page_size))
    raw_events = cache.zrangebyscore(
        _events_key(session_id),
        min=float(after_seq) + 0.5,
        max=float("inf"),
    )
    events: List[Dict[str, Any]] = []
    for item in raw_events[:page_size]:
        if isinstance(item, dict):
            events.append(item)
            continue
        if isinstance(item, (bytes, bytearray)):
            item = item.decode("utf-8")
        if isinstance(item, str):
            try:
                parsed = json.loads(item)
            except (TypeError, ValueError):
                continue
            if isinstance(parsed, dict):
                events.append(parsed)
    return events


def snapshot(session_id: str) -> Optional[Dict[str, Any]]:
    meta = get_session(session_id)
    if meta is None:
        return None
    status: Sam3VideoSessionStatus = meta.get("status") or "starting"
    return {
        "session_id": session_id,
        "status": status,
        "last_seq": int(meta.get("last_seq") or 0),
        "last_frame_id": meta.get("last_frame_id"),
        "error_message": meta.get("error_message"),
        "stop_requested": bool(meta.get("stop_requested")),
    }
