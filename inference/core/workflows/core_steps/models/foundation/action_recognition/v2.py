"""Action Recognition v2 — time-domain controls.

Sliding-window action recognition driven by an OpenAI-compatible VLM. Unlike
v1's frame-count-based controls (max_frames, timeout_seconds), v2 takes
time-domain settings (window_seconds, stride_seconds, sample_fps) and derives
internal frame counts from the stream's video metadata.

Per-frame behavior:
  1. Subsample the incoming frame at `sample_fps`. Frames arriving faster than
     this rate are not encoded — saves CPU.
  2. Append to a per-stream rolling buffer that covers the last `window_seconds`
     of video time (older entries are evicted).
  3. If `stride_seconds` of video time has elapsed since the last LLM fire,
     synchronously call the LLM with the current window. Update last_letter.
  4. Always return the most recent letter (and raw response). Frames between
     fires return the cached value so downstream visualization is smooth.

Because the call to the LLM is synchronous, the workflow engine naturally
serializes calls per video stream — there is no internal queue or rate
limiter. If the API is slower than the configured stride, calls happen at
the API's effective rate; predictions trail real-time by the API latency.
"""

import base64
import math
import time
from collections import deque
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import cv2
import numpy as np
from openai import OpenAI
from pydantic import ConfigDict, Field, field_validator

from inference.core.logger import logger
from inference.core.utils.image_utils import encode_image_to_jpeg_bytes
from inference.core.workflows.execution_engine.entities.base import (
    OutputDefinition,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    FLOAT_KIND,
    IMAGE_KIND,
    INTEGER_KIND,
    LANGUAGE_MODEL_OUTPUT_KIND,
    SECRET_KIND,
    STRING_KIND,
    Selector,
)
from inference.core.workflows.prototypes.block import (
    AirGappedAvailability,
    BlockResult,
    WorkflowBlock,
    WorkflowBlockManifest,
)

DEFAULT_FPS_FALLBACK = 30.0
JPEG_QUALITY = 75
MAX_RESOLUTION = 1920
MAX_WINDOW_SECONDS = 60.0
MAX_STRIDE_SECONDS = 60.0
MAX_SAMPLE_FPS = 60.0
HARD_BUFFER_CAP = 200  # absolute upper bound on frames in the buffer

LONG_DESCRIPTION = """
Time-domain sliding-window action recognition via an OpenAI-compatible VLM.

Configure the **window length** (how much real-world video each LLM call
sees), the **stride** between calls (controls overlap and prediction rate),
and the **sample FPS** (how densely frames are sampled inside the window).
Source FPS is auto-detected from the stream's video metadata.

The block's LLM call is synchronous, so per-stream processing is naturally
sequential — no queueing, no rate limiter needed. Predictions trail real-time
by approximately the API latency. For finite videos this is perfect (lossless,
ordered, small end-of-video delay). For live streams with a fast-enough API
this is also fine; if the API is too slow for your stride, calls happen
at the API's actual rate and predictions lag accordingly.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Action Recognition",
            "version": "v2",
            "short_description": (
                "Time-domain sliding-window action recognition via an "
                "OpenAI-compatible VLM."
            ),
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "action recognition",
                "VLM",
                "video understanding",
                "sliding window",
                "Qwen",
                "OpenAI",
                "Together",
            ],
            "ui_manifest": {
                "section": "model",
                "icon": "fal fa-running",
            },
        },
        protected_namespaces=(),
    )
    type: Literal["roboflow_core/action_recognition@v2"]

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Live video frame from the stream.",
        examples=["$inputs.image"],
    )
    prompt: Union[Selector(kind=[STRING_KIND]), str] = Field(
        title="Prompt",
        description=(
            "Task instruction. Should describe the action being detected and "
            "end with the letter options (e.g. 'A: dunk. B: no dunk.')."
        ),
        examples=[
            "Respond with the letter indicating whether a player is dunking. "
            "A: dunk. B: no dunk.",
        ],
        json_schema_extra={"multiline": True, "always_visible": True},
    )
    choices: List[str] = Field(
        default=["A", "B"],
        title="Allowed Letters",
        description="Letters allowed in the model's response (JSON-schema enum).",
        examples=[["A", "B"], ["A", "B", "C", "D"]],
    )
    window_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=1.0,
        title="Window Length (seconds)",
        description=(
            "How much real video time each LLM call sees. "
            "Combined with sample_fps determines frames-per-call."
        ),
        examples=[0.5, 1.0, 2.0],
    )
    stride_seconds: Optional[Union[float, Selector(kind=[FLOAT_KIND])]] = Field(
        default=None,
        title="Stride (seconds)",
        description=(
            "Time between consecutive LLM fires. If unset, defaults to "
            "window_seconds / 2 (50% overlap). Set equal to window_seconds "
            "for no overlap."
        ),
        examples=[None, 0.25, 0.5, 1.0],
    )
    sample_fps: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=5.0,
        title="Sample FPS",
        description=(
            "How densely frames are sampled inside each window. "
            "Frames-per-call = ceil(window_seconds × sample_fps)."
        ),
        examples=[2.0, 5.0, 10.0],
    )
    base_url: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="https://api.together.xyz/v1",
        title="Base URL",
        description="OpenAI-compatible endpoint URL.",
        examples=["https://api.together.xyz/v1", "http://localhost:8000/v1"],
    )
    model_name: Union[Selector(kind=[STRING_KIND]), str] = Field(
        default="Qwen/Qwen3.5-9B",
        title="Model Name",
        description="Model identifier sent to the API.",
        examples=["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-397B-A17B"],
    )
    api_key: Optional[Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str]] = Field(
        default=None,
        title="API Key",
        description="API key for the endpoint.",
        examples=["$inputs.api_key"],
        private=True,
    )
    resolution: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=512,
        title="Frame Resolution",
        description=(
            f"Maximum side length to downsample frames to before encoding "
            f"(64-{MAX_RESOLUTION})."
        ),
        examples=[384, 512, 640],
    )
    max_tokens: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=500,
        title="Max Tokens",
        description=(
            "Output token budget per LLM call. Default 500 leaves room for "
            "Qwen3.5 models' implicit reasoning before the JSON answer."
        ),
        examples=[100, 500, 1000],
    )

    @field_validator("window_seconds")
    @classmethod
    def _v_window_seconds(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not (0.05 <= float(value) <= MAX_WINDOW_SECONDS):
            raise ValueError(
                f"`window_seconds` must be between 0.05 and {MAX_WINDOW_SECONDS}."
            )
        return value

    @field_validator("stride_seconds")
    @classmethod
    def _v_stride_seconds(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, (int, float)) and not (0.01 <= float(value) <= MAX_STRIDE_SECONDS):
            raise ValueError(
                f"`stride_seconds` must be between 0.01 and {MAX_STRIDE_SECONDS}."
            )
        return value

    @field_validator("sample_fps")
    @classmethod
    def _v_sample_fps(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not (0.1 <= float(value) <= MAX_SAMPLE_FPS):
            raise ValueError(f"`sample_fps` must be between 0.1 and {MAX_SAMPLE_FPS}.")
        return value

    @field_validator("choices")
    @classmethod
    def _v_choices(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("`choices` must contain at least one letter.")
        return value

    @field_validator("resolution")
    @classmethod
    def _v_resolution(cls, value: Any) -> Any:
        if isinstance(value, int) and not (64 <= value <= MAX_RESOLUTION):
            raise ValueError(f"`resolution` must be between 64 and {MAX_RESOLUTION}.")
        return value

    @classmethod
    def get_air_gapped_availability(cls) -> AirGappedAvailability:
        return AirGappedAvailability(available=False, reason="requires_internet")

    @classmethod
    def describe_outputs(cls) -> List[OutputDefinition]:
        return [
            OutputDefinition(name="letter", kind=[STRING_KIND]),
            OutputDefinition(name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]),
            OutputDefinition(name="error_status", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


def _compress_frame(numpy_image: np.ndarray, max_side: int) -> bytes:
    h, w = numpy_image.shape[:2]
    if w > max_side or h > max_side:
        scale = min(max_side / w, max_side / h)
        numpy_image = cv2.resize(
            numpy_image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    return encode_image_to_jpeg_bytes(numpy_image, jpeg_quality=JPEG_QUALITY)


def _frame_video_time(image: WorkflowImageData) -> Tuple[float, Optional[float]]:
    """Return (video_time_seconds, source_fps).

    Prefers the stream's video metadata (frame_number / fps). Falls back to
    wall-clock when metadata is absent so the block stays usable on ad-hoc
    inputs and in tests.
    """
    try:
        m = image.video_metadata
        if m is not None and m.fps and m.fps > 0 and m.frame_number is not None:
            return float(m.frame_number) / float(m.fps), float(m.fps)
    except Exception:
        pass
    return time.monotonic(), None


def _parse_letter(raw: Optional[str], choices: List[str]) -> Optional[str]:
    """Pull the letter out of '{"letter": "X"}' or accept a bare letter."""
    if not raw:
        return None
    raw = raw.strip()
    try:
        import json as _json
        obj = _json.loads(raw)
        if isinstance(obj, dict) and "letter" in obj:
            letter = str(obj["letter"]).strip()
            if letter in choices:
                return letter
    except Exception:
        pass
    if raw and raw[0] in choices:
        return raw[0]
    return None


class _StreamState:
    __slots__ = (
        "buffer",
        "last_fire_video_time",
        "last_subsample_video_time",
        "last_letter",
        "last_raw",
    )

    def __init__(self) -> None:
        # buffer: list of (video_time, jpeg_bytes)
        self.buffer: deque = deque(maxlen=HARD_BUFFER_CAP)
        self.last_fire_video_time: Optional[float] = None
        self.last_subsample_video_time: Optional[float] = None
        self.last_letter: str = ""
        self.last_raw: str = ""


class ActionRecognitionBlockV2(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._states: Dict[str, _StreamState] = {}
        self._client_cache: Dict[Tuple[str, str], OpenAI] = {}

    @classmethod
    def get_init_parameters(cls) -> List[str]:
        return []

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"

    def _get_client(self, base_url: str, api_key: str) -> OpenAI:
        cache_key = (base_url, api_key)
        client = self._client_cache.get(cache_key)
        if client is None:
            client = OpenAI(base_url=base_url, api_key=api_key, timeout=120.0)
            self._client_cache[cache_key] = client
        return client

    def _build_extra_body(self, choices: List[str]) -> Dict[str, Any]:
        return {
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "answer",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "letter": {"type": "string", "enum": list(choices)},
                        },
                        "required": ["letter"],
                        "additionalProperties": False,
                    },
                },
            },
        }

    def run(
        self,
        image: WorkflowImageData,
        prompt: str,
        choices: List[str],
        window_seconds: float,
        stride_seconds: Optional[float],
        sample_fps: float,
        base_url: str,
        model_name: str,
        api_key: Optional[str],
        resolution: int,
        max_tokens: int,
    ) -> BlockResult:
        window_seconds = max(0.05, min(float(window_seconds), MAX_WINDOW_SECONDS))
        sample_fps = max(0.1, min(float(sample_fps), MAX_SAMPLE_FPS))
        resolution = max(64, min(int(resolution), MAX_RESOLUTION))
        max_tokens = max(1, int(max_tokens))

        # Default stride = window/2 (50% overlap) if unset.
        if stride_seconds is None:
            stride_seconds = window_seconds / 2.0
        stride_seconds = max(0.01, min(float(stride_seconds), MAX_STRIDE_SECONDS))

        try:
            video_id = image.video_metadata.video_identifier
        except Exception:
            video_id = "default"

        state = self._states.get(video_id)
        if state is None:
            state = _StreamState()
            self._states[video_id] = state

        now_t, _source_fps = _frame_video_time(image)

        # 1) Subsample on every incoming frame: only encode if sample_fps interval has elapsed.
        sample_interval = 1.0 / sample_fps
        if (
            state.last_subsample_video_time is None
            or (now_t - state.last_subsample_video_time) >= sample_interval
        ):
            jpeg = _compress_frame(image.numpy_image, max_side=resolution)
            state.buffer.append((now_t, jpeg))
            state.last_subsample_video_time = now_t

        # 2) Prune buffer to the last `window_seconds` of video time.
        cutoff = now_t - window_seconds
        while state.buffer and state.buffer[0][0] < cutoff:
            state.buffer.popleft()

        # 3) Fire LLM if stride has elapsed and buffer has anything to send.
        error_status = ""
        should_fire = (
            len(state.buffer) > 0
            and (
                state.last_fire_video_time is None
                or (now_t - state.last_fire_video_time) >= stride_seconds
            )
        )
        if should_fire:
            state.last_fire_video_time = now_t
            jpegs = [j for _t, j in state.buffer]
            try:
                letter, raw = self._fire_llm(
                    jpegs=jpegs,
                    prompt=prompt,
                    choices=choices,
                    base_url=base_url,
                    model_name=model_name,
                    api_key=api_key,
                    max_tokens=max_tokens,
                )
                state.last_raw = raw
                if letter:
                    state.last_letter = letter
            except Exception as e:
                logger.warning(
                    f"Action Recognition v2 LLM call failed: {e}", exc_info=True
                )
                error_status = str(e)

        return {
            "letter": state.last_letter,
            "output": state.last_raw,
            "error_status": error_status,
        }

    def _fire_llm(
        self,
        jpegs: List[bytes],
        prompt: str,
        choices: List[str],
        base_url: str,
        model_name: str,
        api_key: Optional[str],
        max_tokens: int,
    ) -> Tuple[Optional[str], str]:
        content: List[dict] = []
        for jpg in jpegs:
            url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})

        client = self._get_client(base_url, api_key or "no-key")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
            temperature=0,
            extra_body=self._build_extra_body(choices),
        )
        if resp.choices is None or len(resp.choices) == 0:
            raise RuntimeError(f"LLM returned no choices: {resp!r}")
        raw = resp.choices[0].message.content or ""
        letter = _parse_letter(raw, choices)
        return letter, raw
