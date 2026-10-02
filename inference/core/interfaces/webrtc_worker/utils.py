import logging
import time
from typing import Any, Dict, List

from streamvision.webrtc_worker.utils import (  # noqa: F401
    detect_image_output,
    get_cv2_rotation_code,
    get_frame_from_workflow_output,
    get_video_fps,
    get_video_rotation,
    overlay_text_on_np_frame,
    parse_video_file_chunk,
    process_frame,
    rotate_video_frame,
    warmup_cuda,
)

from inference.core import logger
from inference.core.cache import cache
from inference.core.cache.redis import RedisCache
from inference.core.utils.roboflow import get_model_id_chunks
from inference.models.aliases import resolve_roboflow_model_alias
from inference.usage_tracking.collector import usage_collector

logging.getLogger("aiortc").setLevel(logging.WARNING)


def workflow_contains_instant_model(workflow_specification: Dict[str, Any]):
    for step in workflow_specification["steps"]:
        step_type = step["type"]
        if "roboflow_core/roboflow_object_detection_model" in step_type:
            if "model_id" not in step:
                continue
            model_id = step["model_id"]
            model_id = resolve_roboflow_model_alias(model_id=model_id)
            _, version_id = get_model_id_chunks(model_id=model_id)
            if version_id is None:
                return True
    return False


def workflow_contains_preloaded_model(
    workflow_specification: Dict[str, Any], preload_models: List[str]
):
    preload_models = set(preload_models)
    for step in workflow_specification["steps"]:
        if "model_id" not in step:
            continue
        model_id = step["model_id"]
        resolved_model_id = resolve_roboflow_model_alias(model_id=model_id)
        if model_id in preload_models or resolved_model_id in preload_models:
            return True
    return False


def is_over_quota(api_key: str) -> bool:
    api_key_plan_details = usage_collector._plan_details.get_api_key_plan(
        api_key=api_key
    )
    is_over_quota = api_key_plan_details.get(
        usage_collector._plan_details._over_quota_col_name
    )
    return is_over_quota


def _get_concurrent_sessions_key(workspace_id: str) -> str:
    """Get the Redis key for tracking concurrent sessions for a workspace."""
    return f"webrtc:concurrent_sessions:{workspace_id}"


def register_webrtc_session(workspace_id: str, session_id: str) -> None:
    """Register a new concurrent WebRTC session for a workspace.

    Adds the session to a Redis sorted set with current timestamp as score.
    Expired entries are cleaned up on read via ZREMRANGEBYSCORE (O(log N + M)).

    Args:
        workspace_id: The workspace identifier
        session_id: Unique identifier for this session
    """
    if not isinstance(cache, RedisCache):
        logger.warning(
            "[REDIS] Redis not available (cache is %s), skipping session registration",
            type(cache).__name__,
        )
        return

    key = _get_concurrent_sessions_key(workspace_id)
    try:
        cache.client.zadd(key, {session_id: time.time()})
        cache.client.expire(key, 600)  # TTL 600 seconds, extended on each heartbeat
        logger.info(
            "Registered session: workspace=%s, session=%s",
            workspace_id,
            session_id,
        )
    except Exception as e:
        logger.error("Failed to register session: %s", e)


def deregister_webrtc_session(workspace_id: str, session_id: str) -> None:
    """Remove a WebRTC session from the concurrent sessions set.

    Should be called when a session ends to immediately free the quota slot,
    rather than waiting for TTL expiry.

    Args:
        workspace_id: The workspace identifier
        session_id: The session identifier to remove
    """
    if not isinstance(cache, RedisCache):
        logger.warning(
            "[REDIS] Redis not available (cache is %s), skipping session deregistration",
            type(cache).__name__,
        )
        return

    key = _get_concurrent_sessions_key(workspace_id)
    try:
        result = cache.client.zrem(key, session_id)
        logger.info(
            "Deregistered session: workspace=%s, session=%s, removed=%s",
            workspace_id,
            session_id,
            result,
        )
    except Exception as e:
        logger.error("Failed to deregister session: %s", e)


def refresh_webrtc_session(workspace_id: str, session_id: str) -> bool:
    """Refresh the timestamp for a concurrent WebRTC session.

    Should be called periodically to keep the session marked as active.
    If not refreshed, the session will be considered expired after TTL.

    Args:
        workspace_id: The workspace identifier
        session_id: The session identifier to refresh

    Returns:
        True if session was refreshed (existed), False otherwise
    """
    logger.debug(
        "[REDIS] refresh_webrtc_session called: workspace=%s, session=%s, cache_type=%s",
        workspace_id,
        session_id,
        type(cache).__name__,
    )
    if not isinstance(cache, RedisCache):
        logger.warning(
            "[REDIS] Redis not available (cache is %s), cannot refresh session",
            type(cache).__name__,
        )
        return False

    key = _get_concurrent_sessions_key(workspace_id)
    timestamp = time.time()
    try:
        # Only refresh sessions that already exist: we want to avoid attacks
        # where an attacker injects arbitrary session IDs via an authenticated
        # heartbeat endpoint
        if cache.client.zscore(key, session_id) is None:
            logger.warning(
                "[REDIS] Session not found: workspace=%s, session=%s",
                workspace_id,
                session_id,
            )
            return False

        cache.client.zadd(key, {session_id: timestamp})
        cache.client.expire(key, 600)  # Extend TTL on each heartbeat
        logger.info(
            "[REDIS] Refreshed session: workspace=%s, session=%s",
            workspace_id,
            session_id,
        )
        return True
    except Exception as e:
        logger.error("[REDIS] Failed to refresh session: %s", e, exc_info=True)
        return False


def get_concurrent_session_count(workspace_id: str, ttl_seconds: int) -> int:
    """Get the count of concurrent sessions for a workspace.

    Cleans up expired entries (older than TTL) before counting.

    Args:
        workspace_id: The workspace identifier
        ttl_seconds: TTL in seconds - entries older than this are considered expired

    Returns:
        Number of concurrent sessions for the workspace
    """
    if not isinstance(cache, RedisCache):
        logger.warning(
            "Redis not available, cannot count concurrent sessions - allowing request"
        )
        return 0

    key = _get_concurrent_sessions_key(workspace_id)
    cutoff = time.time() - ttl_seconds

    try:
        # Step 1: we remove expired entries
        removed = cache.client.zremrangebyscore(key, "-inf", cutoff)
        logger.info("[REDIS] Removed %s expired entries from %s", removed, key)
        # Step 2: we return what is still valid
        count = cache.client.zcard(key)
        return count
    except Exception as e:
        logger.error(
            "[REDIS] Failed to get concurrent session count: %s", e, exc_info=True
        )
        return 0


def is_over_workspace_session_quota(
    workspace_id: str, quota: int, ttl_seconds: int
) -> bool:
    """Check if a workspace has exceeded its concurrent session quota.

    Args:
        workspace_id: The workspace identifier
        quota: Maximum number of concurrent sessions allowed
        ttl_seconds: TTL for considering sessions as active

    Returns:
        True if the workspace has reached or exceeded the quota
    """
    count = get_concurrent_session_count(workspace_id, ttl_seconds)
    logger.info(
        "Workspace %s has %d concurrent sessions (quota: %d)",
        workspace_id,
        count,
        quota,
    )
    return count >= quota


def get_total_concurrent_sessions(ttl_seconds: int) -> int:
    """Get total concurrent WebRTC sessions across all workspaces.

    Args:
        ttl_seconds: TTL in seconds - entries older than this are considered expired

    Returns:
        Total number of active sessions
    """
    if not isinstance(cache, RedisCache):
        logger.warning(
            "[REDIS] Redis not available, cannot count total concurrent sessions"
        )
        return 0

    pattern = "webrtc:concurrent_sessions:*"
    cutoff = time.time() - ttl_seconds
    total = 0

    try:
        for key in cache.client.scan_iter(match=pattern):
            cache.client.zremrangebyscore(key, "-inf", cutoff)
            total += cache.client.zcard(key)
        return total
    except Exception as e:
        logger.error(
            "[REDIS] Failed to get total concurrent sessions: %s", e, exc_info=True
        )
        return 0
