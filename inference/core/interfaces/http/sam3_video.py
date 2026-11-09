"""SAM3 offline video tracking: embed a clip once, then click and propagate on it.

A clip is decoded frame by frame and kept by the SAM3 video tracker under the
caller's id, scoped to the caller's API key. Segment and propagate calls name
the clip by that id and carry every click on it, so the tracker holds no
per-object state between calls: only the clip (its preprocessed frames and
the encoder output of the frames seen so far).

The tracker is an ``inference_models`` model loaded once per worker process
and kept there, with the clips it holds. Requests of one clip therefore have
to reach the same worker process: a deployment that enables these routes runs
one worker process per replica and sends every request of a clip to the same
replica.

Clips arrive and decode through the video input of action recognition
(``inference.core.utils.video_utils``): the same URL policy, size cap,
probing and sequential decoding.

The routes and their request, response and stream contract are the same as
the ``/sam3/video/*`` routes of the ``inference_server`` package.
"""

import asyncio
import hashlib
import threading
import time
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np
import orjson
from fastapi import HTTPException
from starlette.concurrency import run_in_threadpool

from inference.core.entities.requests.sam3_video import (
    Sam3VideoEmbedRequest,
    Sam3VideoPropagateRequest,
    Sam3VideoSegmentRequest,
)
from inference.core.entities.responses.sam3_video import (
    Sam3VideoEmbedResponse,
    Sam3VideoFrameMasks,
    Sam3VideoPropagateResponse,
    Sam3VideoSegmentResponse,
)
from inference.core.env import (
    ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
    ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
    MODELS_CACHE_AUTH_ENABLED,
    OFFLINE_MODE,
)
from inference.core.exceptions import (
    InputImageLoadError,
    PayloadTooLargeError,
    RequestDataContradiction,
    RoboflowAPINotAuthorizedError,
)
from inference.core.interfaces.http.error_handlers import with_route_exceptions
from inference.core.registries.roboflow import _check_if_api_key_has_access_to_model
from inference.core.roboflow_api import (
    ModelEndpointType,
    get_extra_weights_provider_headers,
)
from inference.core.utils.video_utils import (
    probe_video,
    read_frame_windows,
    video_source_path,
)

SAM3_VIDEO_MODEL_CLASS = "SAM3TrackerVideo"
# A proxy in front of the server may end a response that sends nothing for a
# while, often after 60 to 100 s, so a stream says it is alive more often.
PROPAGATE_HEARTBEAT_SECONDS = 15.0


class Sam3VideoTracker:
    """The worker's one SAM3 video tracker, loaded on first use and kept.

    The tracker holds the embedded clips, so it is never unloaded by the
    model manager. A request naming another model id loads that model in its
    place, which drops every clip the previous one held.

    Args:
        model_manager: The server's model manager; its content-addressed
            artifact cache is shared with the tracker's loading.
    """

    def __init__(self, model_manager: Any):
        self._model_manager = model_manager
        self._lock = threading.Lock()
        self._model_id: Optional[str] = None
        self._model: Any = None

    def get(
        self,
        model_id: str,
        *,
        api_key: Optional[str],
        countinference: Optional[bool] = None,
        service_secret: Optional[str] = None,
    ) -> Any:
        """The tracker for ``model_id``, loading it when it is not held yet.

        The API key is checked against the model on every call when
        ``MODELS_CACHE_AUTH_ENABLED`` is set, as ``ModelManager.add_model``
        does for the other routes.

        Args:
            model_id: Model the request names.
            api_key: Key of the request.
            countinference: Passed to the weights provider.
            service_secret: Passed to the weights provider.

        Returns:
            The loaded ``SAM3TrackerVideo`` model.

        Raises:
            RoboflowAPINotAuthorizedError: When the key has no access to the model.
            RequestDataContradiction: When the model is not a SAM3 video tracker.
        """
        if MODELS_CACHE_AUTH_ENABLED and not OFFLINE_MODE:
            if not _check_if_api_key_has_access_to_model(
                api_key=api_key,
                model_id=model_id,
                endpoint_type=ModelEndpointType.CORE_MODEL,
                countinference=countinference,
                service_secret=service_secret,
            ):
                raise RoboflowAPINotAuthorizedError(
                    f"API key does not have access to model {model_id}"
                )

        with self._lock:
            if self._model_id == model_id:
                return self._model

            model = self._load(
                model_id,
                api_key=api_key,
                countinference=countinference,
                service_secret=service_secret,
            )
            ensure_sam3_video_model(model_id, model)
            self._model, self._model_id = model, model_id

        return model

    def _load(
        self,
        model_id: str,
        *,
        api_key: Optional[str],
        countinference: Optional[bool],
        service_secret: Optional[str],
    ) -> Any:
        from inference_models import AutoModel

        model = AutoModel.from_pretrained(
            model_id_or_path=model_id,
            api_key=api_key,
            allow_untrusted_packages=ALLOW_INFERENCE_MODELS_UNTRUSTED_PACKAGES,
            allow_direct_local_storage_loading=ALLOW_INFERENCE_MODELS_DIRECTLY_ACCESS_LOCAL_PACKAGES,
            weights_provider_extra_headers=get_extra_weights_provider_headers(
                countinference=countinference, service_secret=service_secret
            ),
            content_addressed_artifact_cache=(
                self._model_manager.content_addressed_artifact_cache
            ),
        )

        return model


def ensure_sam3_video_model(model_id: str, model: Any) -> None:
    """Refuse a model that is not the SAM3 video tracker.

    Args:
        model_id: Model id as the caller named it, for the message.
        model: The loaded model.

    Raises:
        RequestDataContradiction: 400 naming the class the model loads as instead.
    """
    class_names = [cls.__name__ for cls in type(model).__mro__]
    if SAM3_VIDEO_MODEL_CLASS not in class_names:
        raise RequestDataContradiction(
            f"Model {model_id!r} loads as {type(model).__name__}; the SAM3 video "
            f"routes need a {SAM3_VIDEO_MODEL_CLASS} model."
        )


def scoped_video_id(video_id: str, api_key: Optional[str]) -> str:
    """The caller's clip id, prefixed so ids of different API keys never meet.

    The prefix is the one the ``inference_server`` package gives client ids.

    Args:
        video_id: The caller's id of the clip.
        api_key: Key of the request.

    Returns:
        The id the tracker keeps the clip under.
    """
    namespace = hashlib.sha256((api_key or "anonymous").encode()).hexdigest()[:16]
    scoped = f"{namespace}:{video_id}"

    return scoped


def read_clip_frames(path: str, *, max_frames: int) -> List[np.ndarray]:
    """Decode every frame of a clip, in order.

    Args:
        path: File holding the clip.
        max_frames: Most frames a clip may have: the frames the tracker's
            clip cache holds.

    Returns:
        The RGB ``HWC`` ``uint8`` frames.

    Raises:
        InputImageLoadError: 400 when the clip does not decode or has fewer
            than two frames.
        PayloadTooLargeError: 413 when it has more than ``max_frames``, from
            the header's frame count before anything is decoded.
    """
    limit = (
        f"the most a clip may have is {max_frames}, which is what "
        "SAM3_VIDEO_CACHE_MAX_GB holds."
    )
    frame_count = probe_video(path)[1]
    if frame_count > max_frames:
        too_many = f"The video has {frame_count} frames; {limit}"
        raise PayloadTooLargeError(message=too_many, public_message=too_many)

    # The header's count is unverified: decode one window one frame longer
    # than the cap, so decoding stops there either way.
    frames = next(read_frame_windows(path, windows=[range(max_frames + 1)]), [])
    if not frames:
        message = "The video has no frames."
        raise InputImageLoadError(message=message, public_message=message)
    if len(frames) > max_frames:
        too_many = f"The video has more than {max_frames} frames; {limit}"
        raise PayloadTooLargeError(message=too_many, public_message=too_many)

    return frames


def embed_video(
    model: Any, *, inference_request: Sam3VideoEmbedRequest
) -> Sam3VideoEmbedResponse:
    """Decode the request's clip and keep it with the tracker under its id.

    Args:
        model: The SAM3 video tracker.
        inference_request: The embed request.

    Returns:
        The clip's id, frame count and frame size.
    """
    started = time.perf_counter()
    video = inference_request.video
    with video_source_path(video.type, video.value) as path:
        frames = read_clip_frames(
            path, max_frames=model.video_cache_limits()["max_frames"]
        )

    described = model.embed_video(
        frames,
        video_id=scoped_video_id(inference_request.video_id, inference_request.api_key),
        encode_frames=inference_request.encode_frames,
    )
    response = Sam3VideoEmbedResponse(
        video_id=inference_request.video_id,
        n_frames=described["n_frames"],
        height=described["height"],
        width=described["width"],
        time=time.perf_counter() - started,
    )

    return response


def segment_video_frame(
    model: Any, *, inference_request: Sam3VideoSegmentRequest
) -> Sam3VideoSegmentResponse:
    """Masks on one frame of an embedded clip for the objects clicked there.

    Args:
        model: The SAM3 video tracker.
        inference_request: The segment request.

    Returns:
        The frame's masks by object id.
    """
    started = time.perf_counter()
    result = model.segment_video_frame(
        scoped_video_id(inference_request.video_id, inference_request.api_key),
        prompts=[prompt.model_dump() for prompt in inference_request.prompts],
        frame_idx=inference_request.frame_idx,
    )
    response = Sam3VideoSegmentResponse(
        frame_idx=result["frame_idx"],
        masks=result["masks"],
        time=time.perf_counter() - started,
    )

    return response


def propagate_video(
    model: Any, *, inference_request: Sam3VideoPropagateRequest
) -> Sam3VideoPropagateResponse:
    """Track every clicked object over an embedded clip, forward and backward.

    Args:
        model: The SAM3 video tracker.
        inference_request: The propagate request.

    Returns:
        The masks of every returned frame, in frame order.
    """
    started = time.perf_counter()
    results = model.propagate_video(
        scoped_video_id(inference_request.video_id, inference_request.api_key),
        prompts=[prompt.model_dump() for prompt in inference_request.prompts],
        frame_indices=inference_request.frame_indices,
    )
    response = Sam3VideoPropagateResponse(
        frames=[Sam3VideoFrameMasks(**frame) for frame in results],
        time=time.perf_counter() - started,
    )

    return response


def stream_propagation(
    model: Any, *, inference_request: Sam3VideoPropagateRequest
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

    The tracking task is started here, in the route handler, on the request's
    event loop. A caller that disconnects ends the stream; the object being
    tracked finishes in the model, the rest are not tracked.

    Args:
        model: The SAM3 video tracker.
        inference_request: The propagate request.

    Returns:
        The response body, one encoded line at a time.

    Raises:
        RequestDataContradiction: 400 before the stream starts when an object
            has no include point.
    """
    per_object = prompts_by_object(inference_request)
    started = time.perf_counter()
    lines: "asyncio.Queue[Optional[dict]]" = asyncio.Queue()
    tracking = asyncio.ensure_future(
        _track_each_object(
            model,
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
        RequestDataContradiction: 400 when there are no clicks or an object
            has no include point.
    """
    per_object: Dict[int, List[dict]] = {}
    for prompt in inference_request.prompts:
        per_object.setdefault(prompt.obj_id, []).append(prompt.model_dump())
    if not per_object:
        raise RequestDataContradiction("Add at least one include point to track.")

    missing = [
        obj_id
        for obj_id, prompts in per_object.items()
        if not any(prompt["label"] == 1 for prompt in prompts)
    ]
    if missing:
        raise RequestDataContradiction(
            f"Objects {missing} have no include point; every object needs one."
        )

    return per_object


async def _track_each_object(
    model: Any,
    *,
    inference_request: Sam3VideoPropagateRequest,
    per_object: Dict[int, List[dict]],
    lines: "asyncio.Queue[Optional[dict]]",
) -> None:
    video_id = scoped_video_id(inference_request.video_id, inference_request.api_key)
    try:
        for obj_id, prompts in per_object.items():
            results = await run_in_threadpool(
                model.propagate_video,
                video_id,
                prompts=prompts,
                frame_indices=inference_request.frame_indices,
            )
            frames = [
                {"frame_idx": frame["frame_idx"], "mask": frame["masks"][str(obj_id)]}
                for frame in results
            ]
            await lines.put({"type": "object", "obj_id": obj_id, "frames": frames})
    except Exception as error:
        await lines.put(_error_line(error))
        return

    await lines.put(None)


def _error_line(error: Exception) -> dict:
    """The ``error`` line for a failure: the answer the route would have given."""

    def _raise() -> None:
        raise error

    try:
        answer = with_route_exceptions(_raise)()
    except HTTPException as http_error:
        line = {
            "type": "error",
            "status": http_error.status_code,
            "detail": http_error.detail,
        }
        return line

    line = {"type": "error", "status": answer.status_code, **orjson.loads(answer.body)}

    return line


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
