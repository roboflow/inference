import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch

from inference_models.models.base.video_segment_classification import (
    VideoSampling,
    VideoSegmentClassificationModel,
    VideoSegmentClassificationPrediction,
)
from inference_models.models.common.roboflow.model_packages import (
    parse_class_names_file,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import (
    SYSTEM_PROMPT_SENTINEL,
    Cosmos3EdgeReasoner,
)

ZERO_SHOT_MAX_NEW_TOKENS = 256
FINE_TUNE_MAX_NEW_TOKENS = 256
# Fine-tunes train on frames decoded with this cap (roboflow-train#880);
# serving at native resolution is a train/serve skew. Zero-shot keeps
# native pixels: at 360 the base model loses small-object events.
FINE_TUNE_MAX_FRAME_SIDE = 360

ZERO_SHOT_OPEN_VOCABULARY_PROMPT = (
    "Describe the main action in this video clip in one short lowercase "
    "phrase of at most 6 words."
)
FINE_TUNE_SYSTEM_PROMPT = (
    "You are Cosmos 3 Edge, a physical AI reasoning model. Watch the video "
    "carefully and answer with only what is asked."
)
FINE_TUNE_LINE_PATTERN = re.compile(
    r"^\s*(?P<token><\|cls:(?P<label>[^|]+?)\|>)\s*"
    r"<(?P<start>-?\d+(?:\.\d+)?)>\s*"
    r"<(?P<end>-?\d+(?:\.\d+)?)>\s*$"
)

_AFTER_CLASS = "after_class"
_BEFORE_FIRST_TIME = "before_first_time"
_FIRST_INTEGER = "first_integer"
_FIRST_FRACTION_START = "first_fraction_start"
_FIRST_FRACTION = "first_fraction"
_BETWEEN_TIMES = "between_times"
_BEFORE_SECOND_TIME = "before_second_time"
_BEFORE_SECOND_TIME_AFTER_SPACE = "before_second_time_after_space"
_SECOND_INTEGER = "second_integer"
_SECOND_FRACTION_START = "second_fraction_start"
_SECOND_FRACTION = "second_fraction"
_LINE_COMPLETE = "line_complete"
_LINE_COMPLETE_CLOSED = "line_complete_closed"
_EXPECT_CLASS = "expect_class"
_LINE_STATES = (
    _AFTER_CLASS,
    _BEFORE_FIRST_TIME,
    _FIRST_INTEGER,
    _FIRST_FRACTION_START,
    _FIRST_FRACTION,
    _BETWEEN_TIMES,
    _BEFORE_SECOND_TIME,
    _BEFORE_SECOND_TIME_AFTER_SPACE,
    _SECOND_INTEGER,
    _SECOND_FRACTION_START,
    _SECOND_FRACTION,
    _LINE_COMPLETE,
    _LINE_COMPLETE_CLOSED,
)


def _is_token_id(value: Any) -> bool:
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _resolve_class_token_ids(
    tokenizer: Any, class_names: Optional[List[str]]
) -> Optional[Dict[str, int]]:
    """Resolve #880 class tokens, returning ``None`` unless all are present."""
    if tokenizer is None or not class_names:
        return None

    added_vocabulary = {}
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if callable(get_added_vocab):
        try:
            candidate = get_added_vocab()
        except Exception:  # pragma: no cover - third-party tokenizer behavior
            candidate = None
        if isinstance(candidate, dict):
            added_vocabulary = candidate

    convert_tokens_to_ids = getattr(tokenizer, "convert_tokens_to_ids", None)
    unknown_token_id = getattr(tokenizer, "unk_token_id", None)
    result = {}
    for class_name in class_names:
        class_token = f"<|cls:{class_name}|>"
        token_id = added_vocabulary.get(class_token)
        if _is_token_id(token_id):
            result[class_name] = int(token_id)
            continue
        if callable(convert_tokens_to_ids):
            try:
                token_id = convert_tokens_to_ids(class_token)
            except Exception:  # pragma: no cover - third-party tokenizer behavior
                token_id = None
        if not _is_token_id(token_id) or (
            _is_token_id(unknown_token_id) and token_id == unknown_token_id
        ):
            return None
        result[class_name] = int(token_id)
    return result


def _decode_token(tokenizer: Any, token_id: int) -> str:
    try:
        return tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode([token_id])


def _advance_line_state(state: str, text: str) -> Optional[str]:
    """Consume decoded text through the parseable fine-tune line grammar.

    The character-class grammar allows repeated spaces and any positive
    number of integer and fractional digits. This guarantees parseability by
    ``FINE_TUNE_LINE_PATTERN``; it intentionally does not enforce exact
    spacing or exactly two fractional digits.
    """
    for character in text:
        if state == _AFTER_CLASS:
            if character != " ":
                return None
            state = _BEFORE_FIRST_TIME
        elif state == _BEFORE_FIRST_TIME:
            if character == " ":
                continue
            if character != "<":
                return None
            state = _FIRST_INTEGER
        elif state == _FIRST_INTEGER:
            if not character.isascii() or not character.isdigit():
                return None
            state = _FIRST_FRACTION_START
        elif state == _FIRST_FRACTION_START:
            if character.isascii() and character.isdigit():
                continue
            if character != ".":
                return None
            state = _FIRST_FRACTION
        elif state == _FIRST_FRACTION:
            if not character.isascii() or not character.isdigit():
                return None
            state = _BETWEEN_TIMES
        elif state == _BETWEEN_TIMES:
            if character.isascii() and character.isdigit():
                continue
            elif character == ">":
                state = _BEFORE_SECOND_TIME
            else:
                return None
        elif state == _BEFORE_SECOND_TIME:
            if character != " ":
                return None
            state = _BEFORE_SECOND_TIME_AFTER_SPACE
        elif state == _BEFORE_SECOND_TIME_AFTER_SPACE:
            if character == " ":
                continue
            if character != "<":
                return None
            state = _SECOND_INTEGER
        elif state == _SECOND_INTEGER:
            if not character.isascii() or not character.isdigit():
                return None
            state = _SECOND_FRACTION_START
        elif state == _SECOND_FRACTION_START:
            if character.isascii() and character.isdigit():
                continue
            if character != ".":
                return None
            state = _SECOND_FRACTION
        elif state == _SECOND_FRACTION:
            if not character.isascii() or not character.isdigit():
                return None
            state = _LINE_COMPLETE
        elif state == _LINE_COMPLETE:
            if character.isascii() and character.isdigit():
                continue
            elif character == ">":
                state = _LINE_COMPLETE_CLOSED
            else:
                return None
        elif state == _LINE_COMPLETE_CLOSED:
            if character != "\n":
                return None
            state = _EXPECT_CLASS
        else:
            return None
    return state


class _FineTunePrefixAllowedTokensFn:
    """Token-level prefix constraint for the #880 answer grammar."""

    def __init__(self, tokenizer: Any, class_token_ids: Dict[str, int]):
        self._class_token_ids = class_token_ids
        self._allowed_class_token_ids = set(class_token_ids.values())
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        self._eos_token_id = int(eos_token_id) if _is_token_id(eos_token_id) else None

        vocabulary_ids: Set[int] = set(self._allowed_class_token_ids)
        get_vocab = getattr(tokenizer, "get_vocab", None)
        if callable(get_vocab):
            try:
                vocabulary = get_vocab()
            except Exception:  # pragma: no cover - third-party tokenizer behavior
                vocabulary = None
            if isinstance(vocabulary, dict):
                vocabulary_ids.update(
                    int(token_id)
                    for token_id in vocabulary.values()
                    if _is_token_id(token_id)
                )
        if self._eos_token_id is not None:
            vocabulary_ids.add(self._eos_token_id)

        normal_token_ids = vocabulary_ids - self._allowed_class_token_ids
        if self._eos_token_id is not None:
            normal_token_ids.discard(self._eos_token_id)
        token_text = {}
        for token_id in normal_token_ids:
            try:
                decoded = _decode_token(tokenizer=tokenizer, token_id=token_id)
            except Exception:  # pragma: no cover - third-party tokenizer behavior
                continue
            if decoded:
                token_text[token_id] = decoded

        # Tokens can combine characters (for example, " <" or ".00>"). A
        # transition is cached only if every decoded character is valid from
        # the current state.
        self._line_transitions = {
            state: {
                token_id: next_state
                for token_id, decoded in token_text.items()
                if (
                    next_state := _advance_line_state(state=state, text=decoded)
                )
                is not None
            }
            for state in _LINE_STATES
        }
        self._none_transitions = {
            prefix: {
                token_id: prefix + decoded
                for token_id, decoded in token_text.items()
                if "none".startswith(prefix + decoded)
            }
            for prefix in ("", "n", "no", "non")
        }
        self._input_length = 0

    def for_classes(self, class_names: List[str]) -> "_FineTunePrefixAllowedTokensFn":
        bound = _FineTunePrefixAllowedTokensFn.__new__(
            _FineTunePrefixAllowedTokensFn
        )
        bound.__dict__ = self.__dict__.copy()
        bound._allowed_class_token_ids = {
            self._class_token_ids[class_name] for class_name in class_names
        }
        bound._input_length = 0
        return bound

    def set_input_length(self, input_length: int) -> None:
        self._input_length = input_length

    def __call__(self, batch_id: int, input_ids: torch.Tensor) -> List[int]:
        del batch_id
        token_ids = input_ids.tolist()
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        token_ids = token_ids[self._input_length :]

        mode = "start"
        state = ""
        for token_id in token_ids:
            if mode == "start":
                if token_id in self._allowed_class_token_ids:
                    mode, state = "line", _AFTER_CLASS
                    continue
                next_prefix = self._none_transitions[""].get(token_id)
                if next_prefix is None:
                    return []
                mode, state = "none", next_prefix
            elif mode == "none":
                if state == "none":
                    return []
                next_prefix = self._none_transitions[state].get(token_id)
                if next_prefix is None:
                    return []
                state = next_prefix
            elif state == _EXPECT_CLASS:
                if token_id not in self._allowed_class_token_ids:
                    return []
                state = _AFTER_CLASS
            else:
                next_state = self._line_transitions.get(state, {}).get(token_id)
                if next_state is None:
                    return []
                state = next_state

        if mode == "start":
            return sorted(
                self._allowed_class_token_ids
                | set(self._none_transitions[""].keys())
            )
        if mode == "none":
            if state == "none":
                return [self._eos_token_id] if self._eos_token_id is not None else []
            return sorted(self._none_transitions[state].keys())
        if state == _EXPECT_CLASS:
            return sorted(self._allowed_class_token_ids)

        result = set(self._line_transitions.get(state, {}).keys())
        if state == _LINE_COMPLETE_CLOSED and self._eos_token_id is not None:
            result.add(self._eos_token_id)
        return sorted(result)


def _normalize_condensed_label(label: str) -> str:
    return " ".join(label.replace("_", " ").lower().split())


def _seconds_to_frame_indices(
    start_seconds: float,
    end_seconds: float,
    num_frames: int,
    fps: float,
) -> Optional[Tuple[int, int]]:
    if (
        num_frames <= 0
        or fps <= 0
        or not math.isfinite(fps)
        or not math.isfinite(start_seconds)
        or not math.isfinite(end_seconds)
    ):
        return None

    max_frame_idx = num_frames - 1
    start_frame_idx = math.floor(start_seconds * fps)
    end_frame_idx = math.ceil(end_seconds * fps)
    start_frame_idx = min(max(start_frame_idx, 0), max_frame_idx)
    end_frame_idx = min(max(end_frame_idx, 0), max_frame_idx)
    if start_frame_idx > end_frame_idx:
        start_frame_idx, end_frame_idx = end_frame_idx, start_frame_idx
    return start_frame_idx, end_frame_idx


def _parse_fine_tune_segments(
    text: str,
    class_names: List[str],
    num_frames: int,
    fps: float,
) -> List[VideoSegmentClassificationPrediction]:
    if not isinstance(text, str) or text.strip().lower() == "none":
        return []

    allowed_classes = set(class_names)
    result = []
    for line in text.splitlines():
        match = FINE_TUNE_LINE_PATTERN.match(line)
        if match is None or match.group("label") not in allowed_classes:
            continue
        frame_indices = _seconds_to_frame_indices(
            start_seconds=float(match.group("start")),
            end_seconds=float(match.group("end")),
            num_frames=num_frames,
            fps=fps,
        )
        if frame_indices is None:
            continue
        result.append(
            VideoSegmentClassificationPrediction(
                start_frame_idx=frame_indices[0],
                end_frame_idx=frame_indices[1],
                class_name=match.group("label"),
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


def _read_package_json(model_name_or_path: str, file_name: str) -> Optional[dict]:
    try:
        with open(Path(model_name_or_path) / file_name) as config_file:
            config = json.load(config_file)
    except (FileNotFoundError, json.JSONDecodeError):
        return None
    return config if isinstance(config, dict) else None


def _read_class_names_file(model_name_or_path: str) -> Optional[List[str]]:
    """Read the package class list, one name per line, in class-token order."""
    class_names_path = Path(model_name_or_path) / "class_names.txt"
    if not class_names_path.is_file():
        return None
    return parse_class_names_file(class_names_path=str(class_names_path)) or None


_CLASS_TOKEN_PATTERN = re.compile(r"^<\|cls:(?P<label>[^|]+?)\|>$")


def _class_names_from_tokenizer(tokenizer: Any) -> Optional[List[str]]:
    """Derive the trained class list from the added <|cls:...|> tokens.

    Token-id order keeps class ids stable across loads.
    """
    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    if not callable(get_added_vocab):
        return None
    try:
        added_vocabulary = get_added_vocab()
    except Exception:  # pragma: no cover - third-party tokenizer behavior
        return None
    if not isinstance(added_vocabulary, dict):
        return None
    class_tokens = sorted(
        (token_id, match.group("label"))
        for token, token_id in added_vocabulary.items()
        if _is_token_id(token_id)
        and (match := _CLASS_TOKEN_PATTERN.match(token)) is not None
    )
    return [label for _, label in class_tokens] or None


def _read_video_pre_processing(model_name_or_path: str) -> Optional[Dict[str, Any]]:
    inference_config = _read_package_json(
        model_name_or_path, "inference_config.json"
    )
    if inference_config is None:
        return None
    video_pre_processing = inference_config.get("video_pre_processing")
    return video_pre_processing if isinstance(video_pre_processing, dict) else None


class Cosmos3EdgeVideoSegmentClassification(VideoSegmentClassificationModel):
    def __init__(
        self,
        reasoner: Cosmos3EdgeReasoner,
        class_names: Optional[List[str]] = None,
        video_pre_processing: Optional[Dict[str, Any]] = None,
    ):
        self._reasoner = reasoner
        self._class_names = class_names
        self._video_pre_processing = video_pre_processing

        processor = getattr(reasoner, "_processor", None)
        tokenizer = getattr(processor, "tokenizer", None)
        self._fine_tune_class_token_ids = _resolve_class_token_ids(
            tokenizer=tokenizer,
            class_names=self.class_names,
        )
        self._fine_tune_prefix_allowed_tokens_fn = (
            _FineTunePrefixAllowedTokensFn(
                tokenizer=tokenizer,
                class_token_ids=self._fine_tune_class_token_ids,
            )
            if self._fine_tune_class_token_ids is not None
            else None
        )

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names

    @property
    def video_pre_processing(self) -> Optional[Dict[str, Any]]:
        return self._video_pre_processing

    @property
    def video_sampling(self) -> VideoSampling:
        config = self._video_pre_processing or {}
        default = VideoSampling()

        def _positive_number(key: str, fallback: float) -> float:
            value = config.get(key)
            if isinstance(value, (int, float)) and value > 0:
                return float(value)
            return fallback

        return VideoSampling(
            window_seconds=_positive_number(
                "window_seconds", default.window_seconds
            ),
            sample_fps=_positive_number("sample_fps", default.sample_fps),
            min_frames=int(_positive_number("min_frames", default.min_frames)),
        )

    @classmethod
    def from_pretrained(
        cls, model_name_or_path: str, **kwargs
    ) -> "Cosmos3EdgeVideoSegmentClassification":
        reasoner = Cosmos3EdgeReasoner.from_pretrained(model_name_or_path, **kwargs)
        class_names = _read_class_names_file(model_name_or_path)
        if class_names is None:
            class_names = _class_names_from_tokenizer(
                getattr(reasoner._processor, "tokenizer", None)
            )
        return cls(
            reasoner=reasoner,
            class_names=class_names,
            video_pre_processing=_read_video_pre_processing(model_name_or_path),
        )

    def infer(
        self,
        frames: List[Union[np.ndarray, torch.Tensor]],
        class_names: Optional[List[str]] = None,
        fps: Optional[float] = None,
        **kwargs,
    ) -> List[VideoSegmentClassificationPrediction]:
        if fps is None:
            raise ValueError("fps is required for video segment classification")

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
    ) -> List[VideoSegmentClassificationPrediction]:
        assert self.class_names is not None
        max_frame_side = FINE_TUNE_MAX_FRAME_SIDE
        if self._video_pre_processing is not None:
            configured_side = self._video_pre_processing.get("max_frame_side")
            if isinstance(configured_side, (int, float)) and configured_side > 0:
                max_frame_side = int(configured_side)
        frames = _cap_frame_side(frames=frames, max_side=max_frame_side)
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
        prompt = (
            f"{user_prompt}{SYSTEM_PROMPT_SENTINEL}{FINE_TUNE_SYSTEM_PROMPT}"
        )
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
        return _parse_fine_tune_segments(
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
    ) -> List[VideoSegmentClassificationPrediction]:
        vocabulary = class_names or self.class_names or None
        if vocabulary is not None and len(vocabulary) > 25:
            raise ValueError("Cosmos3 zero-shot mode supports at most 25 classes")

        if vocabulary is None:
            prompt = ZERO_SHOT_OPEN_VOCABULARY_PROMPT
        else:
            choices = " ".join(
                f"({chr(ord('A') + index)}) {class_name}"
                for index, class_name in enumerate(vocabulary)
            )
            none_letter = chr(ord("A") + len(vocabulary))
            prompt = (
                f"What happens in this video clip? {choices} "
                f"({none_letter}) none of the above. Answer with the letter only."
            )

        generation_kwargs = dict(kwargs)
        generation_kwargs.setdefault("max_new_tokens", ZERO_SHOT_MAX_NEW_TOKENS)
        generation_kwargs.update(
            enable_thinking=False,
            prefix_allowed_tokens_fn=None,
        )
        response = self._reasoner.prompt_video(
            frames=frames,
            prompt=prompt,
            input_color_format="rgb",
            video_fps=fps,
            **generation_kwargs,
        )
        answer = _answer_text(response)
        if vocabulary is not None:
            match = re.search(r"[A-Za-z]", answer)
            if match is None:
                return []
            class_index = ord(match.group(0).upper()) - ord("A")
            if class_index < 0 or class_index >= len(vocabulary):
                return []
            label = vocabulary[class_index]
        else:
            first_line = answer.splitlines()[0] if answer.splitlines() else ""
            label = _normalize_condensed_label(first_line)
            if not label or len(label.split()) > 8:
                return []

        return [
            VideoSegmentClassificationPrediction(
                start_frame_idx=0,
                end_frame_idx=len(frames) - 1,
                class_name=label,
            )
        ]
