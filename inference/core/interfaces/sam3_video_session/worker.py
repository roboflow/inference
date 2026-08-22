from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterable, List, Optional, Protocol

import cv2
import numpy as np
import requests

from inference.core.exceptions import InputImageLoadError
from inference.core.interfaces.sam3_video_session.entities import (
    CHUNK_SAMPLE_SIZE,
    DEFAULT_CLASS_NAME,
    DEFAULT_THRESHOLD,
    SAM3_VIDEO_EVENT_CHECKPOINTED,
    SAM3_VIDEO_EVENT_DONE,
    SAM3_VIDEO_EVENT_DOWNLOADING,
    SAM3_VIDEO_EVENT_ERROR,
    SAM3_VIDEO_EVENT_FRAME,
    MAX_VIDEO_BYTES,
    Sam3VideoTimeBase,
    Sam3VideoWorkerPayload,
)
from inference.core.interfaces.sam3_video_session import predictions as sam3
from inference.core.interfaces.sam3_video_session.artifacts import ArtifactWriter, TrackAccumulator
from inference.core.logger import logger
from inference.core.utils.image_utils import download_url_to_file
from inference.core.utils.requests import api_key_safe_raise_for_status
from inference.core.utils.url_utils import wrap_url


class EventPublisher(Protocol):
    def publish(
        self, event_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Return True when the worker should stop after the current checkpoint."""


class HttpEventPublisher:
    def __init__(
        self,
        *,
        events_callback_url: str,
        publish_token: str,
        api_key: Optional[str],
        timeout_seconds: float = 15.0,
    ):
        self._events_callback_url = events_callback_url
        self._publish_token = publish_token
        self._api_key = api_key
        self._timeout_seconds = timeout_seconds

    def publish(
        self, event_type: str, payload: Optional[Dict[str, Any]] = None
    ) -> bool:
        event: Dict[str, Any] = {"type": event_type}
        if payload:
            event.update(payload)
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        response = requests.post(
            wrap_url(self._events_callback_url),
            json={"publish_token": self._publish_token, "event": event},
            headers=headers,
            timeout=self._timeout_seconds,
        )
        api_key_safe_raise_for_status(response=response)
        try:
            body = response.json()
        except ValueError:
            return False
        return bool(body.get("stop_requested"))


@dataclass
class FramePacket:
    image_bgr: np.ndarray
    frame_index: int
    pts: int
    time_base: float
    source_time_seconds: float
    width: int
    height: int


def time_base_from_fps(fps: float) -> Sam3VideoTimeBase:
    if fps <= 0:
        return Sam3VideoTimeBase(numerator=1, denominator=30)
    return Sam3VideoTimeBase(numerator=1, denominator=max(1, int(round(fps))))


def iter_frames_from_video(
    video_path: str,
    video_time_base: Optional[Sam3VideoTimeBase],
) -> Iterable[FramePacket]:
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise RuntimeError("Could not open the downloaded video for tracking.")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        resolved = video_time_base or time_base_from_fps(fps)
        time_base = resolved.numerator / resolved.denominator
        frame_index = 0
        while True:
            ok, image_bgr = capture.read()
            if not ok or image_bgr is None:
                break
            height, width = image_bgr.shape[:2]
            source_time_seconds = (
                frame_index / fps if fps > 0 else frame_index * time_base
            )
            pts = (
                int(round(source_time_seconds / time_base))
                if time_base
                else frame_index
            )
            yield FramePacket(
                image_bgr=image_bgr,
                frame_index=frame_index,
                pts=pts,
                time_base=time_base,
                source_time_seconds=source_time_seconds,
                width=int(width),
                height=int(height),
            )
            frame_index += 1
    finally:
        capture.release()


def process_frames(
    *,
    frames: Iterable[FramePacket],
    model: Any,
    class_names: List[str],
    threshold: float,
    revision_id: str,
    publisher: EventPublisher,
    writer: ArtifactWriter,
    video_time_base: Sam3VideoTimeBase,
    chunk_sample_size: int = CHUNK_SAMPLE_SIZE,
) -> None:
    accumulators: Dict[str, TrackAccumulator] = {}
    state_dict = None
    produced = 0
    stop_requested = False

    for packet in frames:
        result = sam3.infer_frame(
            model,
            packet.image_bgr,
            class_names,
            state_dict,
        )
        state_dict = result.state_dict
        predictions, samples = sam3.serialize_frame_predictions(
            masks=result.masks,
            object_ids=result.object_ids,
            scores=result.scores,
            boxes=result.boxes,
            prompt_to_object_ids=result.prompt_to_object_ids,
            threshold=threshold,
            width=packet.width,
            height=packet.height,
        )
        for sample in samples:
            sample["frameIndex"] = packet.frame_index
            sample["pts"] = packet.pts
            sample["timeBase"] = packet.time_base
            sample["timestampUs"] = int(round(packet.source_time_seconds * 1_000_000))
            track_id = str(sample["trackId"])
            accumulator = accumulators.get(track_id)
            if accumulator is None:
                accumulator = TrackAccumulator(
                    track_id=track_id,
                    class_name=str(sample["className"]),
                    tracker_id=int(sample["trackId"]),
                )
                accumulators[track_id] = accumulator
            accumulator.add_sample(sample, packet.frame_index, packet.pts)
        flushed = 0
        for accumulator in accumulators.values():
            flushed += accumulator.flush_ready(
                writer,
                chunk_sample_size=chunk_sample_size,
                is_final=False,
            )
        if flushed:
            publisher.publish(
                SAM3_VIDEO_EVENT_CHECKPOINTED,
                {"chunk_count": flushed},
            )
        stop_requested = publisher.publish(
            SAM3_VIDEO_EVENT_FRAME,
            {
                "frame_id": packet.frame_index,
                "revision_id": revision_id,
                "serialized_output_data": {"predictions": {"predictions": predictions}},
                "video_metadata": {
                    "frame_id": packet.frame_index,
                    "pts": packet.pts,
                    "time_base": packet.time_base,
                    "source_time_seconds": packet.source_time_seconds,
                    "width": packet.width,
                    "height": packet.height,
                    "revision_id": revision_id,
                },
            },
        )
        produced += 1
        if stop_requested:
            break

    for accumulator in accumulators.values():
        accumulator.flush_ready(
            writer,
            chunk_sample_size=chunk_sample_size,
            is_final=True,
        )
        accumulator.commit(writer, video_time_base)

    if produced == 0:
        publisher.publish(
            SAM3_VIDEO_EVENT_ERROR,
            {"message": "Tracker produced no frames for this video."},
        )
        return
    if not accumulators:
        publisher.publish(
            SAM3_VIDEO_EVENT_ERROR,
            {"message": "Tracker produced no predictions for this video."},
        )
        return
    publisher.publish(
        SAM3_VIDEO_EVENT_DONE,
        {"cancelled": bool(stop_requested), "frame_count": produced},
    )


def run_sam3_video_session(
    payload: Sam3VideoWorkerPayload,
    *,
    publisher: Optional[EventPublisher] = None,
    writer: Optional[ArtifactWriter] = None,
    model: Any = None,
    frames: Optional[Iterable[FramePacket]] = None,
    chunk_sample_size: int = CHUNK_SAMPLE_SIZE,
) -> None:
    class_names = [
        name.strip() for name in payload.class_names if name and name.strip()
    ]
    if not class_names:
        class_names = [DEFAULT_CLASS_NAME]
    event_publisher = publisher or HttpEventPublisher(
        events_callback_url=payload.events_callback_url,
        publish_token=payload.publish_token,
        api_key=payload.api_key,
    )
    artifact_writer = writer or ArtifactWriter(
        app_base_url=payload.artifact.app_base_url,
        video_id=payload.artifact.video_id,
        workspace_id=payload.artifact.workspace_id,
        dataset_id=payload.artifact.dataset_id,
        revision_id=payload.artifact.revision_id,
        api_key=payload.api_key or "",
    )
    try:
        event_publisher.publish(SAM3_VIDEO_EVENT_DOWNLOADING)
        if frames is None:
            with TemporaryDirectory(prefix="sam3-video-session-") as temp_dir:
                video_path = str(Path(temp_dir) / "video.mp4")
                download_url_to_file(
                    payload.video_url,
                    video_path,
                    max_bytes=MAX_VIDEO_BYTES,
                )
                loaded_model = model or sam3.load_model(payload.api_key)
                if payload.artifact.video_time_base is not None:
                    resolved_time_base = payload.artifact.video_time_base
                else:
                    probe = cv2.VideoCapture(video_path)
                    try:
                        resolved_time_base = time_base_from_fps(
                            float(probe.get(cv2.CAP_PROP_FPS) or 0.0)
                        )
                    finally:
                        probe.release()
                process_frames(
                    frames=iter_frames_from_video(
                        video_path,
                        payload.artifact.video_time_base,
                    ),
                    model=loaded_model,
                    class_names=class_names,
                    threshold=(
                        payload.threshold if payload.threshold else DEFAULT_THRESHOLD
                    ),
                    revision_id=payload.artifact.revision_id,
                    publisher=event_publisher,
                    writer=artifact_writer,
                    video_time_base=resolved_time_base,
                    chunk_sample_size=chunk_sample_size,
                )
            return
        loaded_model = model or sam3.load_model(payload.api_key)
        resolved_time_base = payload.artifact.video_time_base or Sam3VideoTimeBase(
            numerator=1, denominator=30
        )
        process_frames(
            frames=frames,
            model=loaded_model,
            class_names=class_names,
            threshold=payload.threshold if payload.threshold else DEFAULT_THRESHOLD,
            revision_id=payload.artifact.revision_id,
            publisher=event_publisher,
            writer=artifact_writer,
            video_time_base=resolved_time_base,
            chunk_sample_size=chunk_sample_size,
        )
    except InputImageLoadError as error:
        logger.warning("SAM3 video session download failed: %s", error)
        event_publisher.publish(
            SAM3_VIDEO_EVENT_ERROR,
            {"message": error.get_public_error_details()},
        )
    except Exception as error:
        logger.exception("SAM3 video session failed")
        event_publisher.publish(
            SAM3_VIDEO_EVENT_ERROR,
            {"message": str(error) or "Video tracking failed"},
        )


def run_sam3_video_session_from_dict(payload: Dict[str, Any]) -> None:
    run_sam3_video_session(Sam3VideoWorkerPayload.model_validate(payload))
