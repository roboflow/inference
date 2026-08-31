import json
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional, Union

import cv2
import numpy as np
import torch

from inference_models.errors import CorruptedModelPackageError
from inference_models.logger import LOGGER
from inference_models.models.base.action_recognition import (
    SLIDING_WINDOW_MODE,
    WHOLE_VIDEO_MODE,
    ActionRecognitionModel,
    ActionRecognitionPrediction,
    VideoSampling,
)
from inference_models.models.common.roboflow.model_packages import (
    parse_class_names_file,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import (
    SYSTEM_PROMPT_SENTINEL,
    Cosmos3EdgeReasoner,
)
from inference_models.models.cosmos3.span_format import (
    SpanConstrainedDecoder,
    parse_spans,
    resolve_class_token_ids,
)

# Think plus one JSON entry per event outgrows a small budget.
ZERO_SHOT_MAX_NEW_TOKENS = 4096
FINE_TUNE_MAX_NEW_TOKENS = 256

# https://github.com/NVIDIA/cosmos/blob/main/cookbooks/cosmos3/reasoner/reasoner_prompt_guide.md#temporal-localization
# Only this trained phrasing gives dense output. Reworded variants collapse
# to one whole-clip segment, and the reasoning pass is what keeps it dense:
# with thinking off, every trained temporal format lands on round numbers
# that do not follow the content.
ZERO_SHOT_TEMPORAL_LOCALIZATION_PROMPT = """List all action segments in the video.

Provide the result in json format with 'seconds' for time depiction for each event. Use keywords 'start', 'end' and 'caption' in the json output. Please list multiple events if applicable.

```json
[
{
  "start": t_start,
  "end": t_end,
  "caption": EVENT1
},
{
  "start": t_start,
  "end": t_end,
  "caption": EVENT2
},
...
]
```"""
FINE_TUNE_SYSTEM_PROMPT = (
    "You are Cosmos 3 Edge, a physical AI reasoning model. Watch the video "
    "carefully and answer with only what is asked."
)


def _parse_seconds(value: Any) -> Optional[float]:
    """Return a finite JSON number as seconds, else ``None``."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _parse_cookbook_segments(
    text: str,
    num_frames: int,
    fps: float,
) -> List[ActionRecognitionPrediction]:
    """Read the cookbook's second-based JSON list into frame-index ranges."""
    if (
        not isinstance(text, str)
        or num_frames <= 0
        or fps <= 0
        or not math.isfinite(fps)
    ):
        return []
    decoder = json.JSONDecoder()
    entries = None
    # The reasoning pass writes prose around the list, so the first array
    # that decodes is the answer.
    for match in re.finditer(r"\[", text):
        try:
            value, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(value, list):
            entries = value
            break
    if entries is None:
        return []
    max_frame_idx = num_frames - 1
    duration = num_frames / fps
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        label_value = entry.get("caption")
        if not isinstance(label_value, str):
            continue
        label = label_value.strip()
        if not label:
            continue
        start_seconds = _parse_seconds(entry.get("start"))
        end_seconds = _parse_seconds(entry.get("end"))
        if start_seconds is None or end_seconds is None:
            continue
        if end_seconds < 0 or start_seconds > duration:
            continue
        start_frame_idx = min(max(math.floor(start_seconds * fps), 0), max_frame_idx)
        end_frame_idx = min(max(math.ceil(end_seconds * fps), 0), max_frame_idx)
        if start_frame_idx > end_frame_idx:
            start_frame_idx, end_frame_idx = end_frame_idx, start_frame_idx
        result.append(
            ActionRecognitionPrediction(
                start_frame_idx=start_frame_idx,
                end_frame_idx=end_frame_idx,
                class_name=label,
            )
        )
    return result


def _cap_frame_side(frames: List[np.ndarray], max_side: int) -> List[np.ndarray]:
    resized_frames = []
    for frame in frames:
        height, width = frame.shape[:2]
        scale = max_side / max(height, width)
        if scale >= 1.0:
            resized_frames.append(frame)
            continue
        resized_frames.append(
            cv2.resize(
                frame,
                (round(width * scale), round(height * scale)),
                interpolation=cv2.INTER_AREA,
            )
        )
    return resized_frames


def _normalize_frames(
    frames: List[Union[np.ndarray, torch.Tensor]],
) -> List[np.ndarray]:
    normalized_frames = []
    for frame in frames:
        if isinstance(frame, torch.Tensor):
            frame = frame.detach().cpu().permute(1, 2, 0)
            if frame.dtype == torch.bfloat16:
                frame = frame.float()
            frame = frame.numpy()
        normalized_frames.append(frame)
    return normalized_frames


def _answer_text(response: Any) -> str:
    if isinstance(response, dict):
        response = response.get("answer", "")
    return response if isinstance(response, str) else ""


def _has_class_tokens(tokenizer: Any) -> bool:
    """Whether the tokenizer carries fine-tune class tokens."""
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_added_vocab):
        return False
    try:
        added_vocabulary = get_added_vocab()
    except Exception:  # pragma: no cover - third-party tokenizer behavior
        return False
    if not isinstance(added_vocabulary, dict):
        return False
    return any(token.startswith("<|cls:") for token in added_vocabulary)


def _read_class_names_file(model_name_or_path: str) -> Optional[List[str]]:
    """Read the package class list, one name per line, in class-token order."""
    class_names_path = Path(model_name_or_path) / "class_names.txt"
    if not class_names_path.is_file():
        return None
    return parse_class_names_file(class_names_path=str(class_names_path)) or None


def _read_video_sampling(model_name_or_path: str) -> VideoSampling:
    """Read the sampling the training run recorded, or the zero-shot default."""
    default = VideoSampling()
    try:
        with open(Path(model_name_or_path) / "inference_config.json") as config_file:
            inference_config = json.load(config_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return default
    config = (
        inference_config.get("video_pre_processing")
        if isinstance(inference_config, dict)
        else None
    )
    if not isinstance(config, dict):
        return default

    def _positive(key: str, fallback: float) -> float:
        value = config.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
        return fallback

    mode = config.get("mode", default.mode)
    if mode not in (SLIDING_WINDOW_MODE, WHOLE_VIDEO_MODE):
        raise CorruptedModelPackageError(
            message=(
                f"Model package {model_name_or_path} declares video sampling mode "
                f"{mode!r}, which this version of inference does not support. "
                f"Sampling a video the wrong way changes what the model reads, "
                f"so the package is rejected rather than served under a guess."
            ),
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    return VideoSampling(
        window_seconds=_positive("window_seconds", default.window_seconds),
        sample_fps=_positive("sample_fps", default.sample_fps),
        min_frames=int(_positive("min_frames", default.min_frames)),
        max_frame_side=int(_positive("max_frame_side", default.max_frame_side)),
        mode=mode,
    )


class Cosmos3EdgeActionRecognition(ActionRecognitionModel):
    def __init__(
        self,
        reasoner: Cosmos3EdgeReasoner,
        class_names: Optional[List[str]] = None,
        video_sampling: Optional[VideoSampling] = None,
    ):
        self._reasoner = reasoner
        self._class_names = class_names
        self._video_sampling = video_sampling or VideoSampling()

        processor = getattr(reasoner, "_processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        self._fine_tune_class_token_ids = resolve_class_token_ids(
            tokenizer=tokenizer,
            class_names=self.class_names,
        )
        self._fine_tune_prefix_allowed_tokens_fn = (
            SpanConstrainedDecoder(
                tokenizer=tokenizer,
                class_token_ids=self._fine_tune_class_token_ids,
            )
            if self._fine_tune_class_token_ids is not None
            else None
        )
        if self._fine_tune_prefix_allowed_tokens_fn is None:
            # Only training fixes a window. Zero-shot reads a whole clip in
            # one call, which is why it is served over the API and not on a
            # stream that never ends.
            self._video_sampling = replace(self._video_sampling, mode=WHOLE_VIDEO_MODE)

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names

    @property
    def video_sampling(self) -> VideoSampling:
        return self._video_sampling

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "Cosmos3EdgeActionRecognition":
        reasoner = Cosmos3EdgeReasoner.from_pretrained(model_name_or_path, **kwargs)
        tokenizer = getattr(reasoner._processor, "tokenizer", None)
        carries_class_tokens = _has_class_tokens(tokenizer)
        class_names = _read_class_names_file(model_name_or_path)
        if class_names is None and carries_class_tokens:
            # Falling through would serve a fine-tune as a zero-shot model.
            raise CorruptedModelPackageError(
                message=(
                    f"Model package {model_name_or_path} was fine-tuned for "
                    f"action recognition but carries no class_names.txt."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        if class_names and any("|" in class_name for class_name in class_names):
            raise CorruptedModelPackageError(
                message=(
                    f"Model package {model_name_or_path} declares a class name "
                    f"containing '|', which the span format cannot express."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        model = cls(
            reasoner=reasoner,
            class_names=class_names,
            video_sampling=_read_video_sampling(model_name_or_path),
        )
        if (
            class_names or carries_class_tokens
        ) and model._fine_tune_prefix_allowed_tokens_fn is None:
            # Serving these weights zero-shot would answer with the wrong
            # prompt and no decoding constraint, and look like a result.
            raise CorruptedModelPackageError(
                message=(
                    f"Model package {model_name_or_path} lists class names that "
                    f"do not match its class tokens."
                ),
                help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
            )
        return model

    def infer(
        self,
        frames: List[Union[np.ndarray, torch.Tensor]],
        class_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[ActionRecognitionPrediction]:
        if fps is None:
            raise ValueError("fps is required for action recognition")

        normalized_frames = _normalize_frames(frames)
        if not normalized_frames:
            return []
        if self._fine_tune_prefix_allowed_tokens_fn is not None:
            return self._infer_fine_tuned(
                frames=normalized_frames,
                class_filter=class_names,
                fps=fps,
                **kwargs,
            )
        return self._infer_zero_shot(
            frames=normalized_frames,
            class_names=class_names,
            fps=fps,
            **kwargs,
        )

    def _infer_fine_tuned(
        self,
        frames: List[np.ndarray],
        class_filter: Optional[List[str]],
        fps: float,
        **kwargs,
    ) -> List[ActionRecognitionPrediction]:
        assert self.class_names is not None
        frames = _cap_frame_side(
            frames=frames, max_side=self._video_sampling.max_frame_side
        )
        if class_filter is None:
            classes = list(self.class_names)
        else:
            model_classes = set(self.class_names)
            classes = []
            for class_name in class_filter:
                if class_name in model_classes and class_name not in classes:
                    classes.append(class_name)
        if not classes:
            return []
        legend = ", ".join(
            f"<|cls:{class_name}|> ({class_name})" for class_name in classes
        )
        user_prompt = (
            f"Find every occurrence of these classes in the video: {legend}. "
            "For each occurrence output one line as `class token <start> <end>`, "
            "with start and end in seconds from the frame timestamps, in order "
            "of start time. Output `none` if none occurs."
        )
        prompt = f"{user_prompt}{SYSTEM_PROMPT_SENTINEL}{FINE_TUNE_SYSTEM_PROMPT}"
        generation_kwargs = dict(kwargs)
        generation_kwargs.update(
            enable_thinking=False,
            max_new_tokens=FINE_TUNE_MAX_NEW_TOKENS,
            prefix_allowed_tokens_fn=(
                self._fine_tune_prefix_allowed_tokens_fn.for_classes(classes)
            ),
            skip_special_tokens=False,
        )
        response = self._reasoner.prompt_video(
            frames=frames,
            prompt=prompt,
            input_color_format="rgb",
            video_fps=fps,
            **generation_kwargs,
        )
        # skip_special_tokens=False keeps the class tokens but also keeps
        # the chat end token on the final line, which fails the line grammar.
        answer = _answer_text(response)
        eos_token = getattr(
            getattr(self._reasoner._processor, "tokenizer", None), "eos_token", None
        )
        if isinstance(eos_token, str) and eos_token:
            answer = answer.replace(eos_token, "")
        return parse_spans(
            text=answer,
            class_names=classes,
            num_frames=len(frames),
            fps=fps,
        )

    def _infer_zero_shot(
        self,
        frames: List[np.ndarray],
        class_names: Optional[List[str]],
        fps: float,
        **kwargs,
    ) -> List[ActionRecognitionPrediction]:
        if class_names:
            # The checkpoint ignores a vocabulary stated inside the
            # localization prompt, so constraining it here would only look
            # like it worked.
            LOGGER.warning(
                "Cosmos3 zero-shot action recognition answers in its own "
                "words, so the %d requested class(es) are ignored. Fine-tune "
                "the model to classify into a fixed vocabulary.",
                len(class_names),
            )
        generation_kwargs = dict(kwargs)
        generation_kwargs.setdefault("max_new_tokens", ZERO_SHOT_MAX_NEW_TOKENS)
        generation_kwargs.update(
            # The reasoning pass is what keeps the output dense; without it
            # the model answers with round numbers that ignore the content.
            enable_thinking=True,
            prefix_allowed_tokens_fn=None,
        )
        response = self._reasoner.prompt_video(
            frames=frames,
            prompt=ZERO_SHOT_TEMPORAL_LOCALIZATION_PROMPT,
            input_color_format="rgb",
            video_fps=fps,
            **generation_kwargs,
        )
        return _parse_cookbook_segments(
            text=_answer_text(response),
            num_frames=len(frames),
            fps=fps,
        )
