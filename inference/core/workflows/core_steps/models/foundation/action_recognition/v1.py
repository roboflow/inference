"""Action Recognition block.

Replaces the (Image Stack + Rate Limiter + OpenAI-Compatible LLM) three-block
composition with a single block that buffers frames, throttles LLM calls, and
parses a constrained-letter response — all internally.

Defaults mirror the canonical configuration: subsample to 10 fps, keep the
last 5 frames, 100 ms cooldown for buffer-update / fresh-call attempts, send
to https://api.together.xyz/v1 with Qwen/Qwen3.5-9B, response_format with a
JSON-schema enum for the user-supplied `choices`, thinking disabled.
"""

import base64
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
MAX_FRAMES = 32
MAX_RESOLUTION = 1920

LONG_DESCRIPTION = """
Run action recognition over a rolling window of recent video frames using an
OpenAI-compatible vision model. The block buffers frames internally, throttles
calls to the LLM, and constrains the model's reply to a fixed set of letters
via JSON-schema response_format.

This is a one-block convenience wrapper around the (Image Stack + Rate Limiter
+ OpenAI-Compatible LLM) composition.

## How This Block Works

1. On every input frame: JPEG-encode and append to a per-stream rolling buffer
   (capped at `max_frames`).
2. If less than `timeout_seconds` of video time has elapsed since the last LLM
   call, return the most recent prediction immediately (no API call).
3. Otherwise, send the current buffer + `prompt` to the OpenAI-compatible
   endpoint with a JSON-schema response_format whose `letter` enum is the
   user-supplied `choices` list. Parse the returned letter, cache it, and
   return it.

The block always returns *some* letter once a prediction has been made — even
on throttled frames — so downstream visualization is smooth. On the first few
frames before any LLM call has succeeded, `letter` is empty.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Action Recognition",
            "version": "v1",
            "short_description": (
                "Run action recognition over a rolling window of frames using "
                "an OpenAI-compatible VLM."
            ),
            "long_description": LONG_DESCRIPTION,
            "license": "Apache-2.0",
            "block_type": "model",
            "search_keywords": [
                "action recognition",
                "VLM",
                "video understanding",
                "rolling window",
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
    type: Literal["roboflow_core/action_recognition@v1"]

    image: Selector(kind=[IMAGE_KIND]) = Field(
        title="Input Image",
        description="Live video frame.",
        examples=["$inputs.image"],
    )
    prompt: Union[Selector(kind=[STRING_KIND]), str] = Field(
        title="Prompt",
        description=(
            "Question / instruction sent to the model. End with the letter "
            "options, e.g. 'A: stroke. B: no stroke.'"
        ),
        examples=[
            "Respond with the letter indicating whether a tennis stroke is "
            "being done. A: stroke. B: no stroke.",
        ],
        json_schema_extra={"multiline": True, "always_visible": True},
    )
    choices: List[str] = Field(
        default=["A", "B"],
        title="Allowed Letters",
        description=(
            "Letters the model is allowed to return. Forms the JSON-schema "
            "enum that constrains output."
        ),
        examples=[["A", "B"], ["A", "B", "C", "D"]],
    )
    timeout_seconds: Union[float, Selector(kind=[FLOAT_KIND])] = Field(
        default=0.5,
        title="Cooldown (seconds)",
        description=(
            "Minimum time between consecutive LLM calls. Frames arriving in "
            "the meantime update the buffer but do not trigger a new call."
        ),
        examples=[0.1, 0.5, 1.0],
    )
    max_frames: Union[int, Selector(kind=[INTEGER_KIND])] = Field(
        default=5,
        title="Max Frames per Call",
        description=(
            f"How many recent frames to send to the model on each LLM call "
            f"(1-{MAX_FRAMES})."
        ),
        examples=[5, 10, 15],
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
        description="Model identifier to send.",
        examples=["Qwen/Qwen3.5-9B", "Qwen/Qwen3.5-397B-A17B"],
    )
    api_key: Optional[Union[Selector(kind=[STRING_KIND, SECRET_KIND]), str]] = Field(
        default=None,
        title="API Key",
        description="API key for the endpoint, if required.",
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

    @field_validator("max_frames")
    @classmethod
    def validate_max_frames(cls, value: Any) -> Any:
        if isinstance(value, int) and not (1 <= value <= MAX_FRAMES):
            raise ValueError(f"`max_frames` must be between 1 and {MAX_FRAMES}.")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and float(value) < 0.0:
            raise ValueError("`timeout_seconds` must be non-negative.")
        return value

    @field_validator("resolution")
    @classmethod
    def validate_resolution(cls, value: Any) -> Any:
        if isinstance(value, int) and not (64 <= value <= MAX_RESOLUTION):
            raise ValueError(f"`resolution` must be between 64 and {MAX_RESOLUTION}.")
        return value

    @field_validator("choices")
    @classmethod
    def validate_choices(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("`choices` must contain at least one letter.")
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


def _read_source_fps(image: WorkflowImageData) -> Optional[float]:
    try:
        m = image.video_metadata
        if m is not None and m.fps and m.fps > 0:
            return float(m.fps)
    except Exception:
        pass
    return None


def _frame_video_time(image: WorkflowImageData, source_fps: Optional[float]) -> float:
    if source_fps and source_fps > 0:
        try:
            m = image.video_metadata
            if m is not None and m.frame_number is not None:
                return float(m.frame_number) / source_fps
        except Exception:
            pass
    return time.monotonic()


def _parse_letter(raw: Optional[str], choices: List[str]) -> Optional[str]:
    """LLM output is the JSON string '{"letter": "X"}'. Be defensive."""
    if not raw:
        return None
    raw = raw.strip()
    # Try JSON first.
    try:
        import json as _json
        obj = _json.loads(raw)
        if isinstance(obj, dict) and "letter" in obj:
            letter = str(obj["letter"]).strip()
            if letter in choices:
                return letter
    except Exception:
        pass
    # Fall back to bare-letter (sometimes models comply without the wrapper).
    if raw and raw[0] in choices:
        return raw[0]
    return None


class _State:
    __slots__ = ("buffer", "last_fired_t", "last_letter", "last_raw")

    def __init__(self, maxlen: int) -> None:
        self.buffer: deque = deque(maxlen=maxlen)
        self.last_fired_t: Optional[float] = None
        self.last_letter: str = ""
        self.last_raw: str = ""


class ActionRecognitionBlockV1(WorkflowBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._states: Dict[str, _State] = {}
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
        # `chat_template_kwargs.enable_thinking=false` is always sent. It's a
        # no-op for non-Qwen models; for Qwen3.x it prevents hidden reasoning
        # from eating the max_tokens budget before the JSON-schema reply lands.
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
        timeout_seconds: float,
        max_frames: int,
        base_url: str,
        model_name: str,
        api_key: Optional[str],
        resolution: int,
    ) -> BlockResult:
        max_frames = max(1, min(int(max_frames), MAX_FRAMES))
        resolution = max(64, min(int(resolution), MAX_RESOLUTION))
        timeout_seconds = max(0.0, float(timeout_seconds))

        try:
            video_id = image.video_metadata.video_identifier
        except Exception:
            video_id = "default"

        state = self._states.get(video_id)
        if state is None or state.buffer.maxlen != max_frames:
            old = list(state.buffer) if state else []
            new_state = _State(maxlen=max_frames)
            for jpg in old[-max_frames:]:
                new_state.buffer.append(jpg)
            if state:
                new_state.last_fired_t = state.last_fired_t
                new_state.last_letter = state.last_letter
                new_state.last_raw = state.last_raw
            state = new_state
            self._states[video_id] = state

        # Always update the buffer with the current frame.
        state.buffer.append(_compress_frame(image.numpy_image, max_side=resolution))

        # Decide whether to fire the LLM.
        source_fps = _read_source_fps(image) or DEFAULT_FPS_FALLBACK
        now_t = _frame_video_time(image, source_fps)
        should_fire = (
            state.last_fired_t is None
            or (now_t - state.last_fired_t) >= timeout_seconds
        )

        error_status = ""
        if should_fire:
            state.last_fired_t = now_t
            try:
                letter, raw = self._fire_llm(
                    state=state,
                    prompt=prompt,
                    choices=choices,
                    base_url=base_url,
                    model_name=model_name,
                    api_key=api_key,
                )
                state.last_raw = raw
                if letter:
                    state.last_letter = letter
            except Exception as e:
                logger.warning(f"Action Recognition LLM call failed: {e}", exc_info=True)
                error_status = str(e)

        return {
            "letter": state.last_letter,
            "output": state.last_raw,
            "error_status": error_status,
        }

    def _fire_llm(
        self,
        state: _State,
        prompt: str,
        choices: List[str],
        base_url: str,
        model_name: str,
        api_key: Optional[str],
    ) -> Tuple[Optional[str], str]:
        content: List[dict] = []
        for jpg in state.buffer:
            url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})

        client = self._get_client(base_url, api_key or "no-key")
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=100,
            temperature=0,
            extra_body=self._build_extra_body(choices),
        )
        if resp.choices is None or len(resp.choices) == 0:
            raise RuntimeError(f"LLM returned no choices: {resp!r}")
        raw = resp.choices[0].message.content or ""
        letter = _parse_letter(raw, choices)
        return letter, raw
