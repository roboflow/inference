"""Run the Action Recognition Model block's model calls on a server.

The block keeps its per-video state; only each window's model call goes to
``/infer/action_recognition``, as a short clip at the rate it was sampled. The
endpoint plans windows from the clip's own frame rate and frame count, so a
clip that fits one model window becomes exactly one server window, and the
response's frame indices are the sampled-frame indices a local call returns.

Two bounded differences from a local call remain:

* Half-open (``end_exclusive``) spans come back snapped outward to whole
  samples, because the endpoint reports inclusive frame indices.
* The endpoint rounds a window's duration to whole microseconds, so a model
  that turns seconds back into frames can place a boundary one sample apart.

The encoding is lossy by default, for speed. ``encode_window`` states what
that costs against a lossless clip.
"""

import base64
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence

import cv2
import numpy as np
import requests
from roboflow_workflows.environment import (
    HOSTED_CORE_MODEL_URL,
    LOCAL_INFERENCE_API_URL,
    WORKFLOWS_REMOTE_API_KEY_TRANSPORT,
    WORKFLOWS_REMOTE_API_TARGET,
)

from inference_models.entities import Confidence
from inference_models.models.base.action_recognition import SLIDING_WINDOW_MODE
from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction as ModelActionRecognitionPrediction,
)
from inference_models.models.base.action_recognition import VideoSampling
from inference_sdk.http.entities import ApiKeyTransport, InferenceConfiguration
from inference_sdk.http.utils.requests import deduct_api_key_from_string

ACTION_RECOGNITION_ENDPOINT = "/infer/action_recognition"
REMOTE_REQUEST_TIMEOUT_SECONDS = 120.0
LOSSLESS_CODEC = ("FFV1", ".mkv")
LOSSY_CODEC = ("mp4v", ".mp4")
CONTAINER_FPS_TOLERANCE = 1e-3
_MICROSECONDS = 1_000_000


@dataclass(frozen=True)
class RemoteActionRecognitionModel:
    """Stand in for a loaded model, with ``infer`` sent to a server.

    The block reads the same attributes as from a local model, so its
    sampling, buffering and merging do not change. The attributes other than
    ``model_id`` and ``api_key`` mirror ``ActionRecognitionContract``.
    """

    model_id: str
    api_key: Optional[str]
    class_names: Optional[List[str]]
    video_sampling: VideoSampling
    supports_confidence: bool
    supports_observed_duration: bool
    end_exclusive_spans: bool
    frame_storage_transform: Optional[Callable[[Any], Any]]

    def infer(
        self,
        frames: List[Any],
        *,
        fps: float,
        class_names: Optional[List[str]] = None,
        confidence: Optional[Confidence] = None,
        **kwargs,
    ) -> List[ModelActionRecognitionPrediction]:
        """Classify one window through ``/infer/action_recognition``.

        Args:
            frames: Sampled RGB frames in temporal order (HWC arrays or CHW
                uint8 tensors).
            fps: Sampling rate of the frames.
            class_names: Requested class subset, sent as ``class_filter``.
            confidence: Threshold override, sent only when given.
            **kwargs: Ignored. ``duration_seconds`` is not sent: the endpoint
                derives it from the clip, and the block clamps spans to its
                own frame limit.

        Returns:
            Spans in sampled-frame coordinates, as a local ``infer`` returns.

        Raises:
            ValueError: The window cannot map one to one onto a server window.
            RuntimeError: The server read a different clip than was sent.
            requests.HTTPError: The server rejected the request.
            requests.RequestException: The request did not reach the server.
        """
        frame_count = len(frames)
        _ensure_window_maps_one_to_one(
            frame_count=frame_count, fps=fps, video_sampling=self.video_sampling
        )
        clip = encode_window(frames, fps=fps)

        # The SDK has no client call that takes clip bytes, so this request
        # carries what its calls carry: usage attribution, the billing
        # opt-out of an internal run, and the configured key transport.
        configuration = InferenceConfiguration(
            api_key_transport=WORKFLOWS_REMOTE_API_KEY_TRANSPORT
        )
        transport = configuration.api_key_transport
        payload = {
            **configuration.to_source_parameters(),
            "model_id": self.model_id,
            "video": {
                "type": "base64",
                "value": base64.b64encode(clip).decode("ascii"),
            },
        }
        if transport is not ApiKeyTransport.HEADER:
            payload["api_key"] = self.api_key
        headers = None
        if transport is not ApiKeyTransport.LEGACY and self.api_key is not None:
            headers = {"Authorization": f"Bearer {self.api_key}"}
        query = {
            **configuration.to_source_parameters(include_origin=True),
            **(configuration.to_billing_query_parameters() or {}),
        }
        if class_names:
            payload["class_filter"] = class_names
        if confidence is not None:
            payload["confidence"] = confidence
        api_url = (
            HOSTED_CORE_MODEL_URL
            if WORKFLOWS_REMOTE_API_TARGET == "hosted"
            else LOCAL_INFERENCE_API_URL
        )
        try:
            response = requests.post(
                api_url.rstrip("/") + ACTION_RECOGNITION_ENDPOINT,
                json=payload,
                params=query or None,
                headers=headers,
                timeout=REMOTE_REQUEST_TIMEOUT_SECONDS,
            )
        except requests.RequestException as error:
            # The error text quotes the request URL, service secret included,
            # and it reaches the logs and the block's error_status output.
            raise requests.RequestException(
                deduct_api_key_from_string(value=str(error))
            ) from None
        if response.status_code >= 400:
            # The message reaches the block's error_status output.
            detail = deduct_api_key_from_string(value=response.text[:500])
            raise requests.HTTPError(
                f"Action recognition endpoint returned {response.status_code}: "
                f"{detail}",
                response=response,
            )
        result = response.json()

        _ensure_server_read_the_window(result=result, frame_count=frame_count, fps=fps)
        # The endpoint reports inclusive frames; a half-open span ends one
        # sample later, which merge_window_segments maps back the same way.
        segments = [
            ModelActionRecognitionPrediction(
                start_frame_idx=entry["start_frame_idx"],
                end_frame_idx=entry["end_frame_idx"] + int(self.end_exclusive_spans),
                class_name=entry["class"],
                confidence=entry.get("confidence"),
                end_exclusive=self.end_exclusive_spans,
            )
            for entry in result["timeline"]
        ]

        return segments


def load_remote_action_recognition_model(
    model_id: str,
    *,
    api_key: Optional[str],
) -> RemoteActionRecognitionModel:
    """Build a remote model from the model's contract, without its weights.

    The block needs the sampling contract before its first model call, so it
    cannot wait for a server answer.

    Args:
        model_id: Roboflow model identifier.
        api_key: Key with access to the model.

    Returns:
        A model-shaped object whose ``infer`` calls the server.

    Raises:
        ModelImplementationNotFoundError: The model is not an action
            recognition model that ``inference_models`` can describe.
    """
    # Imported here so an older inference_models fails only REMOTE loads, not
    # the import of the whole block.
    from inference_models import get_action_recognition_contract

    contract = get_action_recognition_contract(model_id, api_key=api_key)
    model = RemoteActionRecognitionModel(
        model_id=model_id,
        api_key=api_key,
        class_names=contract.class_names,
        video_sampling=contract.video_sampling,
        supports_confidence=contract.supports_confidence,
        supports_observed_duration=contract.supports_observed_duration,
        end_exclusive_spans=contract.end_exclusive_spans,
        frame_storage_transform=contract.frame_storage_transform,
    )

    return model


def encode_window(
    frames: Sequence[Any], *, fps: float, lossless: bool = False
) -> bytes:
    """Encode RGB frames as a clip whose container states ``fps``.

    The default is OpenCV's MPEG-4 Part 2 (``mp4v``) at its default bitrate.
    Measured on 48 Cosmos windows, it is 14 times smaller than FFV1 (138 KB
    against 1.9 MB for 32 frames at 360 px) and matches the lossless answer
    in 28 of them. Elsewhere a boundary moves by at most three samples, and
    two short ranges at a window's trailing edge drop out. Higher-fidelity
    MJPEG and VP9 moved as many boundaries for more bytes, and the Linux
    OpenCV wheels cannot write H.264.

    OpenCV's FFmpeg writer crops an odd width or height to even, in both
    codecs, so an odd-sized frame loses its last row or column.

    Args:
        frames: RGB frames of one size, HWC arrays or CHW uint8 tensors.
        fps: Frame rate the container declares.
        lossless: Use FFV1, which decodes to the same pixels (even sizes).

    Returns:
        The encoded clip.

    Raises:
        ValueError: Frames differ in size, or no encoder is available.
    """
    bgr_frames = [_to_bgr(frame) for frame in frames]
    height, width = bgr_frames[0].shape[:2]
    if any(frame.shape[:2] != (height, width) for frame in bgr_frames):
        raise ValueError("A remote window needs frames of one size.")

    codec, suffix = LOSSLESS_CODEC if lossless else LOSSY_CODEC
    handle, path = tempfile.mkstemp(suffix=suffix)
    os.close(handle)
    try:
        writer = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*codec), float(fps), (width, height)
        )
        if not writer.isOpened():
            raise ValueError(f"OpenCV cannot encode {codec} on this host.")
        for frame in bgr_frames:
            writer.write(frame)
        writer.release()
        clip = Path(path).read_bytes()
    finally:
        os.remove(path)

    return clip


def _to_bgr(frame: Any) -> np.ndarray:
    if not isinstance(frame, np.ndarray):
        frame = frame.detach().permute(1, 2, 0).to("cpu").numpy()
    bgr_frame = np.ascontiguousarray(frame[:, :, ::-1])

    return bgr_frame


def _ensure_window_maps_one_to_one(
    *, frame_count: int, fps: float, video_sampling: VideoSampling
) -> None:
    # A silent mismatch between sent and classified frames shifts action
    # boundaries, so refuse any window the endpoint would pad or split.
    if frame_count < 2:
        raise ValueError(
            "The action recognition endpoint needs at least two frames, so a "
            "one-frame window cannot run remotely."
        )
    if frame_count < video_sampling.min_frames:
        raise ValueError(
            f"A window of {frame_count} frames is below the model's minimum of "
            f"{video_sampling.min_frames}, so the endpoint would repeat frames."
        )
    if video_sampling.mode != SLIDING_WINDOW_MODE:
        return

    # The endpoint compares durations in whole microseconds.
    clip_us = int(round(frame_count / fps * _MICROSECONDS))
    window_us = int(round(video_sampling.window_seconds * _MICROSECONDS))
    if clip_us > window_us:
        raise ValueError(
            f"A window of {frame_count} frames at {fps} FPS runs longer than the "
            f"model's {video_sampling.window_seconds} s window, so the endpoint "
            f"would split it and the frames would no longer map one to one."
        )


def _ensure_server_read_the_window(
    *, result: dict, frame_count: int, fps: float
) -> None:
    read_frames = result["frame_count"]
    read_fps = result["source_fps"]
    windows = result["windows_classified"]
    if (
        read_frames != frame_count
        # The clip's container keeps about three decimals of the rate.
        or not math.isclose(read_fps, fps, abs_tol=CONTAINER_FPS_TOLERANCE)
        or windows not in (0, 1)
    ):
        raise RuntimeError(
            f"The server read {read_frames} frames at {read_fps} FPS in {windows} "
            f"windows, but the block sent {frame_count} frames at {fps} FPS in one "
            f"window. The response cannot map back to the sampled frames."
        )
