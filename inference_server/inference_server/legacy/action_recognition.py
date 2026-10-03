"""Action recognition over a clip, one window at a time.

The model declares how a clip is cut and sampled, so a caller sends the clip
and nothing else. The clip is probed and decoded off the event loop; each
window then crosses the model bridge as a params-only call, in clip order,
and its segments are unioned into one timeline that counts the clip's frames.
"""

import asyncio
import contextlib
from typing import Any, List, Optional, Sequence

from inference_models.models.base.action_recognition import (
    VideoSampling,
    effective_max_frame_side,
    merge_segment,
    plan_windows,
)
from inference_server.legacy.bridge import Route
from inference_server.legacy.entities import (
    ActionRecognitionInferenceResponse,
    ActionRecognitionPrediction,
)
from inference_server.legacy.errors import LegacyHTTPError
from inference_server.legacy.video import (
    ensure_clip_fits_the_duration_cap,
    probe_video,
    read_frame_windows,
    video_source_path,
)

ACTION_RECOGNITION_TASK = "action-recognition"
ACTION_RECOGNITION_ACTION = "infer"
_COSMOS_REASONER_CLASS = "Cosmos3EdgeReasoner"


def ensure_action_recognition_route(model_id: str, route: Route) -> None:
    """Refuse a model that is not served for the action-recognition task.

    Args:
        model_id: Model id as the caller named it, for the message.
        route: Resolved route of the model.

    Raises:
        LegacyHTTPError: 400 naming the task the model serves instead. A bare
            Cosmos reasoner, which a platform fine-tune registered under the
            VLM task loads as, is named as such.
    """
    if route.task_type == ACTION_RECOGNITION_TASK:
        return

    if route.model_class_name == _COSMOS_REASONER_CLASS:
        raise LegacyHTTPError(
            400,
            f"Model {model_id!r} is a {route.task_type} model, loaded as "
            f"{_COSMOS_REASONER_CLASS}; action recognition needs a model "
            f"registered for the {ACTION_RECOGNITION_TASK} task.",
        )
    raise LegacyHTTPError(400, f"Model {model_id!r} is a {route.task_type} model.")


async def classify_video(
    route: Route,
    api_key: Optional[str],
    bridge: Any,
    *,
    video_type: str,
    video_value: Any,
    class_filter: Optional[List[str]],
) -> ActionRecognitionInferenceResponse:
    """Classify the actions in a clip with the model behind ``route``.

    The clip lands in a temporary file, is checked against the duration cap
    and cut into the windows the model's sampling asks for. The decoder is one
    sequential pass advanced a window at a time from a worker thread, never
    from two at once, and each window is classified before the next is read,
    so at most the window being classified and the one being read are held. A
    window with fewer frames than the sampling's minimum is skipped. The
    decoder is closed when the windows end; a read still in flight when the
    request is cancelled is awaited to completion first, then the decoder is
    closed and the cancellation propagates.

    Args:
        route: Resolved route of an action recognition model; its metadata
            carries the sampling contract and the class list.
        api_key: Key the model was resolved with.
        bridge: Model bridge offering ``infer_params_only``.
        video_type: ``url`` or ``base64``.
        video_value: The URL, or the base64 text of the clip.
        class_filter: Classes a fine-tuned model should report; ``None`` for
            all of them.

    Returns:
        The classified ranges covering the clip, sorted by start frame and
        class id, without the request's identity fields.

    Raises:
        LegacyHTTPError: 400 for a clip that cannot be fetched or decoded, 413
            for one over the size or duration cap.
    """
    sampling = (
        VideoSampling(**route.video_sampling)
        if route.video_sampling is not None
        else VideoSampling()
    )
    id_vocabulary = route.class_names or None
    timeline: List[ActionRecognitionPrediction] = []
    windows_classified = 0
    async with video_source_path(video_type, video_value) as path:
        source_fps, frame_count = await asyncio.to_thread(probe_video, path)
        ensure_clip_fits_the_duration_cap(frame_count=frame_count, fps=source_fps)
        windows = plan_windows(
            frame_count=frame_count, source_fps=source_fps, sampling=sampling
        )
        window_frames = read_frame_windows(
            path,
            windows=[window.frame_indices for window in windows],
            max_frame_side=effective_max_frame_side(sampling),
        )
        read: Optional["asyncio.Future[List[Any]]"] = None
        try:
            for window in windows:
                read = asyncio.ensure_future(asyncio.to_thread(next, window_frames))
                frames = await asyncio.shield(read)
                read = None
                if len(frames) < max(1, sampling.min_frames):
                    continue

                windows_classified += 1
                segments = await bridge.infer_params_only(
                    route,
                    api_key,
                    ACTION_RECOGNITION_ACTION,
                    {
                        "frames": frames,
                        "class_names": class_filter,
                        "fps": window.sample_fps,
                    },
                )
                merge_window_segments(
                    timeline=timeline,
                    frame_numbers=window.frame_indices[: len(frames)],
                    segments=segments,
                    id_vocabulary=id_vocabulary,
                    stride=max(1.0, source_fps / window.sample_fps),
                )
        finally:
            if read is not None:
                await _settle_read(read)
            window_frames.close()

    timeline.sort(key=lambda entry: (entry.start_frame_idx, entry.class_id))
    response = ActionRecognitionInferenceResponse(
        timeline=timeline,
        source_fps=source_fps,
        frame_count=frame_count,
        windows_classified=windows_classified,
    )

    return response


async def _settle_read(read: "asyncio.Future[List[Any]]") -> None:
    while not read.done():
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait({read})
    if not read.cancelled():
        read.exception()


def merge_window_segments(
    timeline: List[ActionRecognitionPrediction],
    frame_numbers: Sequence[int],
    segments: List[Any],
    id_vocabulary: Optional[List[str]],
    stride: float,
    class_filter: Optional[List[str]] = None,
) -> None:
    """Union one window's segments into ``timeline``, in place.

    ``frame_numbers`` holds the source frame each sampled index came from, in
    the order the model saw them. ``class_filter`` drops classes the caller
    did not ask for. Classes outside ``id_vocabulary`` report ``-1``, which is
    what an open-vocabulary answer gets.

    Args:
        timeline: Ranges found so far, widened or extended in place.
        frame_numbers: Source frame of each sampled index of the window.
        segments: The model's segments, indexed into the window's samples.
        id_vocabulary: The model's own class list, or ``None`` without one.
        stride: Sampling stride in source frames; ranges closer than it merge.
        class_filter: Classes to keep, or ``None`` to keep every class.
    """
    sample_count = len(frame_numbers)
    if sample_count == 0:
        return
    class_ids = (
        {class_name: index for index, class_name in enumerate(id_vocabulary)}
        if id_vocabulary is not None
        else {}
    )
    for segment in segments:
        class_name = segment.class_name
        if class_filter is not None and class_name not in class_filter:
            continue
        start_index = min(sample_count - 1, max(0, int(segment.start_frame_idx)))
        end_index = min(sample_count - 1, max(0, int(segment.end_frame_idx)))
        if start_index > end_index:
            start_index, end_index = end_index, start_index
        merge_segment(
            timeline=timeline,
            segment=ActionRecognitionPrediction(
                start_frame_idx=frame_numbers[start_index],
                end_frame_idx=frame_numbers[end_index],
                class_name=class_name,
                class_id=class_ids.get(class_name, -1),
            ),
            stride=stride,
        )
