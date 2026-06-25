"""Action Recognition — time-domain controls.

Sliding-window action recognition driven by an OpenAI-compatible VLM. Takes
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
import os
import shutil
import subprocess
import time
import uuid
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

# Set to a path to enable on-disk serialization of the exact frame window sent
# to the LLM on every fire. After the response, the clip is kept only if the
# parsed letter is "A"; otherwise it is deleted. Useful for debugging FPs/FNs.
DEBUG_DUMP_DIR = os.environ.get(
    "ACTION_RECOGNITION_DEBUG_DIR", "/app/inference/snap_debug_server"
)

LONG_DESCRIPTION = """
Recognize actions in a **video stream** by asking an OpenAI-compatible
vision-language model (VLM) to classify short, overlapping clips.

Instead of looking at a single frame, the block keeps a rolling **window** of
recent frames (e.g. the last 1 second of video) and periodically sends that
window to the model, which answers with a single letter from a set of choices
you define (e.g. `A: dunk. B: no dunk.`). Use it to detect whether an action is
happening right now in a live feed or recorded video — dunking, falling,
waving, a machine running, etc.

This block is designed for **video workflows**: feed it the per-frame image
from a video stream and it emits a prediction on every frame, so downstream
visualization and logic stay smooth.

## How this block works

1. **Sample** — incoming frames are subsampled to **Sample FPS**. Frames
   arriving faster than this rate are dropped before encoding (saves CPU).
2. **Buffer** — sampled frames are appended to a per-stream rolling buffer that
   always covers the last **Window Length** seconds of video. Source FPS is
   auto-detected from the stream's video metadata.
3. **Fire** — once **Stride** seconds of video time have elapsed since the last
   call, the current window is sent to the model and the predicted letter is
   cached. Stride controls how often the model runs and how much windows
   overlap (default: half the window, i.e. 50% overlap).
4. **Return** — every frame returns the most recent letter. Frames between
   fires return the cached value, so you always get a prediction.

The model call is synchronous, so processing per stream is naturally
sequential — no queueing or rate limiter needed. Predictions trail real time by
roughly the API latency; if the API is slower than your stride, calls simply
happen at the API's effective rate.

## How to use it

1. **Connect the image** input to a video frame (e.g. `$inputs.image`).
2. **Write a prompt** that describes the action and ends with the lettered
   options, e.g. *"Respond with the letter indicating whether a player is
   dunking. A: dunk. B: no dunk."*
3. **List the allowed letters** in *Allowed Letters* (e.g. `["A", "B"]`). The
   model is constrained by a JSON-schema enum to reply with exactly one of them.
4. **Point at a provider** — set *Base URL*, *Model Name*, and *API Key* for any
   OpenAI-compatible endpoint (Together AI by default; also OpenRouter,
   DashScope, vLLM, a local server, etc.).
5. **Tune the timing** — *Window Length*, *Stride*, and *Sample FPS* trade off
   responsiveness, context length, and cost.

Set **Payload Mode** to match your provider: `frames` sends each frame as a
separate image (works with Together AI Qwen models), while `video` encodes the
window to mp4 and sends a single video part (works with OpenRouter, DashScope,
and other backends that accept OpenAI-style `video_url` content).

## Outputs

- `letter` — the predicted choice for the current window (one of *Allowed
  Letters*), repeated on every frame until the next call.
- `output` — the raw text the model returned for the most recent call.
- `error_status` — populated with the error message if a call failed, otherwise
  empty; use it for flow control or debugging.

This block calls an external API, so it **requires internet access** and is not
available in air-gapped deployments.
"""


class BlockManifest(WorkflowBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            "name": "Action Recognition",
            "version": "v1",
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
    type: Literal["roboflow_core/action_recognition@v1"]

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
    payload_mode: Literal["frames", "video"] = Field(
        default="frames",
        title="Payload Mode",
        description=(
            "How the rolling buffer is sent to the API. 'frames' sends each "
            "frame as a separate image_url content block (works with Together "
            "AI Qwen models). 'video' encodes the buffer to mp4 and sends a "
            "single video_url block (works with OpenRouter, DashScope, and "
            "any backend supporting OpenAI-style video_url content)."
        ),
        examples=["frames", "video"],
    )

    @field_validator("window_seconds")
    @classmethod
    def _v_window_seconds(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not (
            0.05 <= float(value) <= MAX_WINDOW_SECONDS
        ):
            raise ValueError(
                f"`window_seconds` must be between 0.05 and {MAX_WINDOW_SECONDS}."
            )
        return value

    @field_validator("stride_seconds")
    @classmethod
    def _v_stride_seconds(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, (int, float)) and not (
            0.01 <= float(value) <= MAX_STRIDE_SECONDS
        ):
            raise ValueError(
                f"`stride_seconds` must be between 0.01 and {MAX_STRIDE_SECONDS}."
            )
        return value

    @field_validator("sample_fps")
    @classmethod
    def _v_sample_fps(cls, value: Any) -> Any:
        if isinstance(value, (int, float)) and not (
            0.1 <= float(value) <= MAX_SAMPLE_FPS
        ):
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
            OutputDefinition(
                name="output", kind=[STRING_KIND, LANGUAGE_MODEL_OUTPUT_KIND]
            ),
            OutputDefinition(name="error_status", kind=[STRING_KIND]),
        ]

    @classmethod
    def get_execution_engine_compatibility(cls) -> Optional[str]:
        return ">=1.4.0,<2.0.0"


def _jpegs_to_mp4_bytes(jpegs: List[bytes], fps: float) -> Optional[bytes]:
    """Encode a list of JPEG frames into an in-memory mp4 (h264) at `fps`.

    Uses ffmpeg via stdin/stdout pipes so nothing hits disk. Returns the mp4
    bytes, or None on failure. Used for providers that accept a single
    `video_url` content block (e.g. OpenRouter, DashScope) instead of N
    individual image_url items.
    """
    if not jpegs or not shutil.which("ffmpeg"):
        return None
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-r",
        str(float(fps)),
        "-i",
        "pipe:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "frag_keyframe+empty_moov",
        "-f",
        "mp4",
        "pipe:1",
    ]
    try:
        proc = subprocess.run(
            cmd, input=b"".join(jpegs), capture_output=True, check=True
        )
        return proc.stdout
    except subprocess.CalledProcessError as e:
        logger.warning(
            f"ffmpeg mp4 encode failed: {e.stderr.decode('utf-8', 'ignore')[:300]}"
        )
        return None
    except Exception as e:
        logger.warning(f"ffmpeg mp4 encode error: {e}")
        return None


def _write_jpegs_to_mp4(jpegs: List[bytes], path: str, fps: float) -> bool:
    """Decode jpegs and serialize them as a video at the given fps.

    Uses MJPG inside an .avi container (most reliable cv2 codec combo in
    headless containers). Returns True on success.
    """
    if not jpegs:
        return False
    first = cv2.imdecode(np.frombuffer(jpegs[0], dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        return False
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), float(fps), (w, h))
    if not writer.isOpened():
        return False
    try:
        writer.write(first)
        for jpg in jpegs[1:]:
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue
            writer.write(frame)
    finally:
        writer.release()
    return True


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
    """Extract the chosen letter from a model response.

    OpenAI-compatible providers do not all honor the `response_format`
    json-schema strictly, so the answer arrives in several shapes. We accept
    all of them rather than only the canonical `{"letter": "X"}`:

      - object with the expected key:   {"letter": "A"}
      - object with a different key:    {"answer": "A"}, {"choice": "A"}
      - bare JSON string:               "A"
      - bare / quoted letter:           A   or   "A"
      - a short sentence:               "The answer is A."

    Matching is case-insensitive and the canonical choice casing is returned.
    """
    if not raw:
        return None
    import json as _json
    import re as _re

    text = raw.strip()
    # Map upper-cased choice -> canonical choice so matching is case-insensitive.
    canonical = {str(c).strip().upper(): c for c in choices}

    def _coerce(value: Any) -> Optional[str]:
        key = str(value).strip().upper()
        return canonical.get(key)

    # 1) JSON payloads: a bare string ("A") or an object holding the answer.
    try:
        obj = _json.loads(text)
    except Exception:
        obj = None
    if isinstance(obj, str):
        hit = _coerce(obj)
        if hit is not None:
            return hit
    elif isinstance(obj, dict):
        if "letter" in obj:
            hit = _coerce(obj["letter"])
            if hit is not None:
                return hit
        for value in obj.values():  # tolerate {"answer": "A"} etc.
            hit = _coerce(value)
            if hit is not None:
                return hit

    # 2) Plain text (incl. a bare letter or a short sentence): match the first
    #    *standalone* choice token. The word boundaries avoid false hits like
    #    the "B" inside "Based".
    pattern = r"\b(" + "|".join(_re.escape(str(c).strip()) for c in choices) + r")\b"
    match = _re.search(pattern, text, flags=_re.IGNORECASE)
    if match:
        return _coerce(match.group(1))
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


class ActionRecognitionBlockV1(WorkflowBlock):
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

    def _build_extra_body(
        self,
        choices: List[str],
        payload_mode: str = "frames",
        sample_fps: float = 5.0,
    ) -> Dict[str, Any]:
        extra: Dict[str, Any] = {
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
        # When sending a single video_url, tell the server's video preprocessor
        # to sample at our sample_fps instead of its default (Qwen3.5: 2 fps).
        # Otherwise the model throws away most of the frames we encoded.
        if payload_mode == "video":
            extra["mm_processor_kwargs"] = {
                "fps": float(sample_fps),
                "do_sample_frames": True,
            }
        return extra

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
        payload_mode: str = "frames",
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
        should_fire = len(state.buffer) > 0 and (
            state.last_fire_video_time is None
            or (now_t - state.last_fire_video_time) >= stride_seconds
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
                    payload_mode=payload_mode,
                    sample_fps=sample_fps,
                    debug_tag=f"vt{now_t:.3f}",
                )
                state.last_raw = raw
                if letter:
                    state.last_letter = letter
            except Exception as e:
                logger.warning(
                    f"Action Recognition LLM call failed: {e}", exc_info=True
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
        payload_mode: str = "frames",
        sample_fps: float = 5.0,
        debug_tag: str = "",
    ) -> Tuple[Optional[str], str]:
        # Pre-build the mp4 if we'll be sending video_url — we also reuse it
        # for the debug dump so the on-disk file is bit-identical to the
        # payload that hit the API.
        mp4_bytes: Optional[bytes] = None
        if payload_mode == "video":
            mp4_bytes = _jpegs_to_mp4_bytes(jpegs, fps=float(sample_fps))
            if mp4_bytes is None:
                raise ValueError(
                    "payload_mode=video requested but mp4 encode failed; "
                    "falling back to frames."
                )

        # Serialize the exact payload to disk *before* the API call. After the
        # response we keep it only if letter == "A", else delete.
        debug_path: Optional[str] = None
        if DEBUG_DUMP_DIR:
            try:
                os.makedirs(DEBUG_DUMP_DIR, exist_ok=True)
                ext = "mp4" if payload_mode == "video" else "avi"
                fname = (
                    f"fire_{int(time.time() * 1000)}_{debug_tag}_"
                    f"{uuid.uuid4().hex[:6]}.{ext}"
                )
                debug_path = os.path.join(DEBUG_DUMP_DIR, fname)
                if payload_mode == "video" and mp4_bytes is not None:
                    with open(debug_path, "wb") as f:
                        f.write(mp4_bytes)
                else:
                    _write_jpegs_to_mp4(jpegs, debug_path, fps=8.0)
            except Exception as e:
                logger.warning(f"Action Recognition debug dump failed: {e}")
                debug_path = None

        content: List[dict] = []
        if payload_mode == "video" and mp4_bytes is not None:
            url = "data:video/mp4;base64," + base64.b64encode(mp4_bytes).decode("ascii")
            content.append({"type": "video_url", "video_url": {"url": url}})
        else:
            for jpg in jpegs:
                url = "data:image/jpeg;base64," + base64.b64encode(jpg).decode("ascii")
                content.append({"type": "image_url", "image_url": {"url": url}})
        content.append({"type": "text", "text": prompt})

        client = self._get_client(base_url, api_key or "no-key")
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": content}],
                max_tokens=max_tokens,
                temperature=0,
                extra_body=self._build_extra_body(
                    choices, payload_mode=payload_mode, sample_fps=sample_fps
                ),
            )
        except Exception:
            if debug_path and os.path.exists(debug_path):
                try:
                    os.unlink(debug_path)
                except Exception:
                    pass
            raise

        if resp.choices is None or len(resp.choices) == 0:
            if debug_path and os.path.exists(debug_path):
                try:
                    os.unlink(debug_path)
                except Exception:
                    pass
            raise RuntimeError(f"LLM returned no choices: {resp!r}")
        raw = resp.choices[0].message.content or ""
        letter = _parse_letter(raw, choices)

        if debug_path and os.path.exists(debug_path):
            if letter == "A":
                # Rename to include the letter so it's easy to grep/inspect.
                stem, ext = os.path.splitext(debug_path)
                final = f"{stem}_A{ext}"
                try:
                    os.rename(debug_path, final)
                except Exception:
                    pass
            else:
                try:
                    os.unlink(debug_path)
                except Exception:
                    pass

        return letter, raw
