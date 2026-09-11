"""The span format a fine-tuned action recognition model answers in.

A fine-tune answers with one line per event, `<|cls:name|> <S.SS> <E.EE>`,
or the single word `none`. This module owns both directions of that format:
:class:`SpanConstrainedDecoder` admits only the tokens that continue a valid
line during generation, and :func:`parse_spans` reads the result back into
frame ranges. They describe one format, so a change to it lands here rather
than in the model wrapper.

The constraint is a token-level state machine rather than a character-level
grammar because training adds each class as its own vocabulary token. At the
class position the legal set is therefore a set of token ids, which a regex
over characters cannot state directly.

:class:`SpanConstrainedDecoder` satisfies the ``prefix_allowed_tokens_fn``
hook that ``Cosmos3EdgeReasoner`` passes to ``generate``, plus the optional
``set_input_length`` the reasoner calls before each run.
"""

import math
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from inference_models.models.base.action_recognition import ActionRecognitionPrediction

LINE_PATTERN = re.compile(
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


def resolve_class_token_ids(
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
    ``LINE_PATTERN``; it intentionally does not enforce exact
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


class SpanConstrainedDecoder:
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
                if (next_state := _advance_line_state(state=state, text=decoded))
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

    def for_classes(self, class_names: List[str]) -> "SpanConstrainedDecoder":
        bound = SpanConstrainedDecoder.__new__(SpanConstrainedDecoder)
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
        # Generation calls this once per token, and the video prompt ahead of
        # the answer runs to thousands of them. Drop the prompt on the tensor
        # so only the answer crosses to host.
        if input_ids.dim() > 1:
            input_ids = input_ids[0]
        token_ids = input_ids[self._input_length :].tolist()

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
                self._allowed_class_token_ids | set(self._none_transitions[""].keys())
            )
        if mode == "none":
            if state == "none":
                return [self._eos_token_id] if self._eos_token_id is not None else []
            return sorted(self._none_transitions[state].keys())
        if state == _EXPECT_CLASS:
            # A trailing newline ends a valid answer, so generation must be
            # able to stop here instead of inventing another span.
            allowed = set(self._allowed_class_token_ids)
            if self._eos_token_id is not None:
                allowed.add(self._eos_token_id)
            return sorted(allowed)

        result = set(self._line_transitions.get(state, {}).keys())
        if state == _LINE_COMPLETE_CLOSED and self._eos_token_id is not None:
            result.add(self._eos_token_id)
        return sorted(result)


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
    if start_seconds > end_seconds:
        start_seconds, end_seconds = end_seconds, start_seconds
    # Clamping first would turn a span the model placed outside the clip into
    # a one-frame event on the nearest edge. The syntax constraint bounds the
    # shape of a line, not the times it names.
    duration_seconds = num_frames / fps
    if end_seconds < 0 or start_seconds > duration_seconds:
        return None
    start_frame_idx = math.floor(start_seconds * fps)
    end_frame_idx = math.ceil(end_seconds * fps)
    start_frame_idx = min(max(start_frame_idx, 0), max_frame_idx)
    end_frame_idx = min(max(end_frame_idx, 0), max_frame_idx)
    if start_frame_idx > end_frame_idx:
        start_frame_idx, end_frame_idx = end_frame_idx, start_frame_idx
    return start_frame_idx, end_frame_idx


def parse_spans(
    text: str,
    class_names: List[str],
    num_frames: int,
    fps: float,
) -> List[ActionRecognitionPrediction]:
    if not isinstance(text, str) or text.strip().lower() == "none":
        return []

    allowed_classes = set(class_names)
    result = []
    for line in text.splitlines():
        match = LINE_PATTERN.match(line)
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
            ActionRecognitionPrediction(
                start_frame_idx=frame_indices[0],
                end_frame_idx=frame_indices[1],
                class_name=match.group("label"),
            )
        )
    return result
