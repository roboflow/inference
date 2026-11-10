"""SAM3 offline video tracking: embed a clip once, then click and propagate on it.

A clip is decoded frame by frame and kept by the SAM3 video tracker under the
caller's id, scoped to the caller's API key like SAM3 image embedding ids.
Segment and propagate calls name the clip by that id and carry every click on
it, so the tracker holds no per-object state between calls: only the clip
(its preprocessed frames and the encoder output of the frames seen so far).
That state lives in one worker, so the routes need the requests of one clip to
reach the same worker.

Clips arrive and decode through the video input of the action recognition
route (``legacy.video``): the same URL policy, size cap, probing and
sequential decoding.
"""

import asyncio
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
import orjson

from inference_model_manager.hash_namespacing import namespace_client_hash_id
from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    Sam3VideoEmbedRequest,
    Sam3VideoEmbedResponse,
    Sam3VideoFrameMasks,
    Sam3VideoPropagateRequest,
    Sam3VideoPropagateResponse,
    Sam3VideoSegmentRequest,
    Sam3VideoSegmentResponse,
)
from inference_server.legacy.errors import LegacyHTTPError, legacy_error_response
from inference_server.legacy.telemetry_recording import record_route_error
from inference_server.legacy.video import (
    probe_video,
    read_frame_windows,
    video_source_path,
)

SAM3_VIDEO_MODEL_CLASS = "SAM3TrackerVideo"
# A proxy in front of the server may end a response that sends nothing for a
# while, often after 60 to 100 s, so a stream says it is alive more often.
PROPAGATE_HEARTBEAT_SECONDS = 15.0


def ensure_sam3_video_route(model_id: str, route: Route) -> None:
    """Refuse a model that is not the SAM3 video tracker.

    Args:
        model_id: Model id as the caller named it, for the message.
        route: Resolved route of the model.

    Raises:
        LegacyHTTPError: 400 naming the class the model loads as instead.
    """
    if route.model_class_name == SAM3_VIDEO_MODEL_CLASS:
        return

    raise LegacyHTTPError(
        400,
        f"Model {model_id!r} loads as {route.model_class_name}; the SAM3 video "
        f"routes need a {SAM3_VIDEO_MODEL_CLASS} model.",
    )


def read_clip_frames(path: str, *, max_frames: int) -> List[np.ndarray]:
    """Decode every frame of a clip, in order.

    Args:
        path: File holding the clip.
        max_frames: Most frames a clip may have: the frames the tracker's
            clip cache holds.

    Returns:
        The RGB ``HWC`` ``uint8`` frames.

    Raises:
        LegacyHTTPError: 400 when the clip does not decode or has fewer than
            two frames, 413 when it has more than ``max_frames`` (from the
            header's frame count, before anything is decoded).
    """
    limit = (
        f"the most a clip may have is {max_frames}, which is what "
        "SAM3_VIDEO_CACHE_MAX_GB holds."
    )
    frame_count = probe_video(path)[1]
    if frame_count > max_frames:
        raise LegacyHTTPError(413, f"The video has {frame_count} frames; {limit}")

    # The header's count is unverified: decode one window one frame longer
    # than the cap, so decoding stops there either way.
    frames = next(read_frame_windows(path, windows=[range(max_frames + 1)]), [])
    if not frames:
        raise LegacyHTTPError(400, "The video has no frames.")
    if len(frames) > max_frames:
        raise LegacyHTTPError(
            413, f"The video has more than {max_frames} frames; {limit}"
        )

    return frames


async def embed_video(
    route: Route,
    api_key: Optional[str],
    bridge: Any,
    *,
    inference_request: Sam3VideoEmbedRequest,
) -> Sam3VideoEmbedResponse:
    """Decode the request's clip and keep it with the tracker under its id.

    Args:
        route: Resolved route of a SAM3 video tracker.
        api_key: Key the model was resolved with; scopes the clip id.
        bridge: Model bridge offering ``infer_params_only``.
        inference_request: The embed request.

    Returns:
        The clip's id, frame count and frame size.

    Raises:
        LegacyHTTPError: 400 for a clip that cannot be fetched or decoded, 413
            for one over the size or frame cap.
    """
    started = time.perf_counter()
    limits = await bridge.infer_params_only(
        route, api_key, "video_cache_limits", {}, record=False
    )
    video = inference_request.video
    async with video_source_path(video.type, video.value) as path:
        frames = await asyncio.to_thread(
            read_clip_frames, path, max_frames=limits["max_frames"]
        )

    described = await bridge.infer_params_only(
        route,
        api_key,
        "embed_video",
        {
            "frames": frames,
            "video_id": namespace_client_hash_id(inference_request.video_id, api_key),
            "encode_frames": inference_request.encode_frames,
            "keep_existing": inference_request.keep_existing,
        },
    )
    response = Sam3VideoEmbedResponse(
        video_id=inference_request.video_id,
        n_frames=described["n_frames"],
        height=described["height"],
        width=described["width"],
        time=time.perf_counter() - started,
    )

    return response


async def segment_video_frame(
    route: Route,
    api_key: Optional[str],
    bridge: Any,
    *,
    inference_request: Sam3VideoSegmentRequest,
) -> Sam3VideoSegmentResponse:
    """Masks on one frame of an embedded clip for the objects clicked there.

    Args:
        route: Resolved route of a SAM3 video tracker.
        api_key: Key the model was resolved with; scopes the clip id.
        bridge: Model bridge offering ``infer_params_only``.
        inference_request: The segment request.

    Returns:
        The frame's masks by object id.
    """
    started = time.perf_counter()
    result = await bridge.infer_params_only(
        route,
        api_key,
        "segment_video_frame",
        {
            "video_id": namespace_client_hash_id(inference_request.video_id, api_key),
            "prompts": [prompt.model_dump() for prompt in inference_request.prompts],
            "frame_idx": inference_request.frame_idx,
        },
    )
    response = Sam3VideoSegmentResponse(
        frame_idx=result["frame_idx"],
        masks=result["masks"],
        time=time.perf_counter() - started,
    )

    return response


async def propagate_video(
    route: Route,
    api_key: Optional[str],
    bridge: Any,
    *,
    inference_request: Sam3VideoPropagateRequest,
) -> Sam3VideoPropagateResponse:
    """Track every clicked object over an embedded clip, forward and backward.

    Args:
        route: Resolved route of a SAM3 video tracker.
        api_key: Key the model was resolved with; scopes the clip id.
        bridge: Model bridge offering ``infer_params_only``.
        inference_request: The propagate request.

    Returns:
        The masks of every returned frame, in frame order.
    """
    started = time.perf_counter()
    results = await bridge.infer_params_only(
        route,
        api_key,
        "propagate_video",
        {
            "video_id": namespace_client_hash_id(inference_request.video_id, api_key),
            "prompts": [prompt.model_dump() for prompt in inference_request.prompts],
            "frame_indices": inference_request.frame_indices,
        },
    )
    response = Sam3VideoPropagateResponse(
        frames=[Sam3VideoFrameMasks(**frame) for frame in results],
        time=time.perf_counter() - started,
    )

    return response


def stream_propagation(
    route: Route,
    api_key: Optional[str],
    bridge: Any,
    *,
    inference_request: Sam3VideoPropagateRequest,
) -> AsyncIterator[bytes]:
    """Track each clicked object over an embedded clip, as NDJSON lines.

    The tracking runs one object per model call, so each object's masks go
    out as soon as that object is tracked. Every line is a JSON object with a
    ``type``:

    - ``object``: ``obj_id`` and ``frames``, its ``frame_idx`` and ``mask``
      (COCO RLE) on every returned frame, in frame order.
    - ``heartbeat``: sent after every ``PROPAGATE_HEARTBEAT_SECONDS`` without
      another line, with the seconds so far as ``time``.
    - ``done``: the last line of a stream that tracked every object, with
      ``time``.
    - ``error``: the last line of a stream that failed after it started, with
      the ``status`` and body the route would have answered with, such as 400
      with ``No video embedded …`` for a clip the worker no longer has.

    The tracking task is started here, in the route handler, so it keeps the
    request's usage and redaction scope. A caller that disconnects ends the
    stream; the object being tracked finishes in the model, the rest are not
    tracked.

    Args:
        route: Resolved route of a SAM3 video tracker.
        api_key: Key the model was resolved with; scopes the clip id.
        bridge: Model bridge offering ``infer_params_only``.
        inference_request: The propagate request.

    Returns:
        The response body, one encoded line at a time.

    Raises:
        LegacyHTTPError: 400 before the stream starts when an object has no
            include point.
    """
    per_object = prompts_by_object(inference_request)
    started = time.perf_counter()
    lines: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()
    tracking = asyncio.ensure_future(
        _track_each_object(
            route,
            api_key,
            bridge,
            inference_request=inference_request,
            per_object=per_object,
            lines=lines,
        )
    )
    body = _ndjson_body(tracking, lines=lines, started=started)

    return body


def prompts_by_object(
    inference_request: Sam3VideoPropagateRequest,
) -> Dict[int, List[dict]]:
    """The request's clicks by object, in order of each object's first click.

    Args:
        inference_request: The propagate request.

    Returns:
        Each object's clicks, in click order.

    Raises:
        LegacyHTTPError: 400 when there are no clicks or an object has no
            include point.
    """
    per_object: Dict[int, List[dict]] = {}
    for prompt in inference_request.prompts:
        per_object.setdefault(prompt.obj_id, []).append(prompt.model_dump())
    if not per_object:
        raise LegacyHTTPError(400, "Add at least one include point to track.")

    missing = [
        obj_id
        for obj_id, prompts in per_object.items()
        if not any(prompt["label"] == 1 for prompt in prompts)
    ]
    if missing:
        raise LegacyHTTPError(
            400, f"Objects {missing} have no include point; every object needs one."
        )

    return per_object


async def _track_each_object(
    route: Route,
    api_key: Optional[str],
    bridge: Any,
    *,
    inference_request: Sam3VideoPropagateRequest,
    per_object: Dict[int, List[dict]],
    lines: "asyncio.Queue[Optional[dict]]",
) -> None:
    video_id = namespace_client_hash_id(inference_request.video_id, api_key)
    try:
        for obj_id, prompts in per_object.items():
            results = await bridge.infer_params_only(
                route,
                api_key,
                "propagate_video",
                {
                    "video_id": video_id,
                    "prompts": prompts,
                    "frame_indices": inference_request.frame_indices,
                },
            )
            frames = [
                {"frame_idx": frame["frame_idx"], "mask": frame["masks"][str(obj_id)]}
                for frame in results
            ]
            await lines.put({"type": "object", "obj_id": obj_id, "frames": frames})
    except Exception as error:
        record_route_error(error)
        answer = legacy_error_response(error)
        await lines.put(
            {
                "type": "error",
                "status": answer.status_code,
                **orjson.loads(answer.body),
            }
        )
        return

    await lines.put(None)


async def _ndjson_body(
    tracking: "asyncio.Future[None]",
    *,
    lines: "asyncio.Queue[Optional[dict]]",
    started: float,
) -> AsyncIterator[bytes]:
    try:
        while True:
            try:
                line = await asyncio.wait_for(
                    lines.get(), timeout=PROPAGATE_HEARTBEAT_SECONDS
                )
            except asyncio.TimeoutError:
                line = {"type": "heartbeat", "time": time.perf_counter() - started}
            if line is None:
                line = {"type": "done", "time": time.perf_counter() - started}
            yield orjson.dumps(line) + b"\n"

            if line["type"] in ("done", "error"):
                return
    finally:
        if not tracking.done():
            tracking.cancel()
