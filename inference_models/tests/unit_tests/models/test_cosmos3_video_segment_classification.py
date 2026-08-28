import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.models.base.video_segment_classification import (
    VideoSegmentClassificationPrediction,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import (
    SYSTEM_PROMPT_SENTINEL,
    Cosmos3EdgeReasoner,
)
from inference_models.models.cosmos3.cosmos3_video_segment_classification import (
    FINE_TUNE_MAX_NEW_TOKENS,
    FINE_TUNE_SYSTEM_PROMPT,
    ZERO_SHOT_MAX_NEW_TOKENS,
    ZERO_SHOT_OPEN_VOCABULARY_PROMPT,
    Cosmos3EdgeVideoSegmentClassification,
    _FineTunePrefixAllowedTokensFn,
    _parse_fine_tune_segments,
)


class _FakeTokenizer:
    def __init__(self, class_names=()):
        ordinary_tokens = [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            ".",
            "<",
            ">",
            " ",
            "\n",
            "n",
            "o",
            "e",
            "none",
            "x",
        ]
        self.unk_token_id = 0
        self.eos_token_id = 99
        self.pad_token_id = 99
        self.eos_token = "<|im_end|>"
        self._vocabulary = {
            token: index + 1 for index, token in enumerate(ordinary_tokens)
        }
        self._added_vocabulary = {
            f"<|cls:{class_name}|>": 50 + index
            for index, class_name in enumerate(class_names)
        }
        self._vocabulary.update(self._added_vocabulary)
        self._id_to_token = {
            token_id: token for token, token_id in self._vocabulary.items()
        }

    def get_added_vocab(self):
        return dict(self._added_vocabulary)

    def get_vocab(self):
        return dict(self._vocabulary)

    def convert_tokens_to_ids(self, token):
        return self._vocabulary.get(token, self.unk_token_id)

    def decode(self, token_ids, **kwargs):
        del kwargs
        return "".join(self._id_to_token.get(token_id, "") for token_id in token_ids)

    def id(self, token):
        return self._vocabulary[token]


class _FakeProcessor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer


class _FakeReasoner:
    def __init__(self, response="", tokenizer=None):
        self.response = response
        self.calls = []
        self._processor = _FakeProcessor(tokenizer or _FakeTokenizer())

    def prompt_video(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

def _frames(count):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    return [frame] * count


def _prediction(start, end, label):
    return VideoSegmentClassificationPrediction(
        start_frame_idx=start,
        end_frame_idx=end,
        class_name=label,
    )


def test_class_token_presence_routes_to_fine_tune_mode_for_long_clip() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner,
        class_names=["walking", "running"],
    )

    result = wrapper.infer(frames=_frames(200), fps=2.0)

    assert result == []
    call = reasoner.calls[0]
    assert call["max_new_tokens"] == FINE_TUNE_MAX_NEW_TOKENS
    assert call["enable_thinking"] is False
    assert call["prefix_allowed_tokens_fn"] is not None
    assert SYSTEM_PROMPT_SENTINEL in call["prompt"]


def test_missing_class_tokens_routes_long_clip_to_zero_shot_mode() -> None:
    reasoner = _FakeReasoner(response="B", tokenizer=_FakeTokenizer())
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner,
        class_names=["walking", "running"],
    )

    result = wrapper.infer(frames=_frames(200), fps=2.0)

    assert result == [_prediction(0, 199, "running")]
    call = reasoner.calls[0]
    assert call["max_new_tokens"] == ZERO_SHOT_MAX_NEW_TOKENS
    assert call["prefix_allowed_tokens_fn"] is None
    assert SYSTEM_PROMPT_SENTINEL not in call["prompt"]


def test_fine_tune_mode_caps_frame_side_at_the_training_resolution() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner, class_names=["walking"]
    )

    large_frames = [np.zeros((480, 854, 3), dtype=np.uint8) for _ in range(4)]
    wrapper.infer(frames=large_frames, fps=2.0)

    sent_frames = reasoner.calls[0]["frames"]
    assert all(frame.shape == (202, 360, 3) for frame in sent_frames)


def test_zero_shot_mode_keeps_native_frame_resolution() -> None:
    reasoner = _FakeReasoner(response="B")
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)

    large_frames = [np.zeros((480, 854, 3), dtype=np.uint8) for _ in range(4)]
    wrapper.infer(frames=large_frames, class_names=["walking"], fps=2.0)

    sent_frames = reasoner.calls[0]["frames"]
    assert all(frame.shape == (480, 854, 3) for frame in sent_frames)


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ("a", [_prediction(0, 3, "walking")]),
        ("B", [_prediction(0, 3, "running")]),
        ("C", []),
        ("no letter here", []),
    ],
)
def test_zero_shot_maps_first_answer_letter_to_class(answer, expected) -> None:
    reasoner = _FakeReasoner(response=answer)
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)

    result = wrapper.infer(
        frames=_frames(4),
        class_names=["walking", "running"],
        fps=5.0,
    )

    assert result == expected
    assert reasoner.calls[0]["prompt"] == (
        "What happens in this video clip? (A) walking (B) running "
        "(C) none of the above. Answer with the letter only."
    )
    assert reasoner.calls[0]["max_new_tokens"] == ZERO_SHOT_MAX_NEW_TOKENS
    assert reasoner.calls[0]["enable_thinking"] is False


def test_zero_shot_uses_caller_max_new_tokens() -> None:
    reasoner = _FakeReasoner(response="A")
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)

    wrapper.infer(
        frames=_frames(2),
        class_names=["walking"],
        fps=5.0,
        max_new_tokens=17,
    )

    assert reasoner.calls[0]["max_new_tokens"] == 17


def test_zero_shot_empty_call_vocabulary_falls_back_to_model_classes() -> None:
    reasoner = _FakeReasoner(response="B")
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner,
        class_names=["walking", "running"],
    )

    result = wrapper.infer(
        frames=_frames(2),
        class_names=[],
        fps=5.0,
    )

    assert result == [_prediction(0, 1, "running")]


def test_zero_shot_rejects_more_than_25_classes() -> None:
    reasoner = _FakeReasoner(response="A")
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)

    with pytest.raises(ValueError, match="at most 25"):
        wrapper.infer(
            frames=_frames(2),
            class_names=[f"class {index}" for index in range(26)],
            fps=5.0,
        )

    assert reasoner.calls == []


def test_zero_shot_open_vocabulary_normalizes_first_line() -> None:
    reasoner = _FakeReasoner(response="Pick_Up   Green_Cup\nextra details")
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)

    result = wrapper.infer(frames=_frames(3), fps=5.0)

    assert result == [_prediction(0, 2, "pick up green cup")]
    assert reasoner.calls[0]["prompt"] == ZERO_SHOT_OPEN_VOCABULARY_PROMPT


@pytest.mark.parametrize(
    "answer",
    [
        "",
        "\nvalid phrase on the second line",
        "one two three four five six seven eight nine",
    ],
)
def test_zero_shot_open_vocabulary_rejects_empty_or_overlong_first_line(
    answer,
) -> None:
    reasoner = _FakeReasoner(response=answer)
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)

    assert wrapper.infer(frames=_frames(3), fps=5.0) == []


def test_fine_tune_prompt_contains_exact_legend_instruction_and_system_prompt() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner,
        class_names=["walking", "running"],
    )

    wrapper.infer(frames=_frames(4), fps=5.0)

    user_prompt = (
        "Find every occurrence of these classes in the video: "
        "<|cls:walking|> (walking), <|cls:running|> (running). "
        "For each occurrence output one line as `class token <start> <end>`, "
        "with start and end in seconds from the frame timestamps, in order "
        "of start time. Output `none` if none occurs."
    )
    assert reasoner.calls[0]["prompt"] == (
        f"{user_prompt}{SYSTEM_PROMPT_SENTINEL}{FINE_TUNE_SYSTEM_PROMPT}"
    )
    assert reasoner.calls[0]["skip_special_tokens"] is False


def test_fine_tune_parser_accepts_valid_lines_and_converts_seconds() -> None:
    result = _parse_fine_tune_segments(
        text=(
            "<|cls:walking|> <-0.20> <0.21>\n"
            "<|cls:running|> <1.40> <0.20>"
        ),
        class_names=["walking", "running"],
        num_frames=10,
        fps=5.0,
    )

    assert result == [
        _prediction(0, 2, "walking"),
        _prediction(1, 7, "running"),
    ]


@pytest.mark.parametrize("answer", ["none", "  NONE  ", "not parseable"])
def test_fine_tune_parser_returns_empty_for_none_or_no_valid_lines(answer) -> None:
    assert (
        _parse_fine_tune_segments(
            text=answer,
            class_names=["walking"],
            num_frames=10,
            fps=5.0,
        )
        == []
    )


def test_fine_tune_parser_drops_out_of_vocabulary_labels() -> None:
    result = _parse_fine_tune_segments(
        text=(
            "<|cls:walking|> <0.00> <0.20>\n"
            "<|cls:jumping|> <0.20> <0.40>"
        ),
        class_names=["walking"],
        num_frames=5,
        fps=5.0,
    )

    assert result == [_prediction(0, 1, "walking")]


def test_fine_tune_class_filter_drops_non_members_and_limits_parser() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(
        response=(
            "<|cls:walking|> <0.00> <0.20>\n"
            "<|cls:running|> <0.20> <0.40>"
        ),
        tokenizer=tokenizer,
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner,
        class_names=["walking", "running"],
    )

    result = wrapper.infer(
        frames=_frames(5),
        class_names=["running", "not-a-model-class"],
        fps=5.0,
    )

    assert result == [_prediction(1, 2, "running")]
    assert "<|cls:running|> (running)" in reasoner.calls[0]["prompt"]
    assert "walking" not in reasoner.calls[0]["prompt"]
    assert "not-a-model-class" not in reasoner.calls[0]["prompt"]


def test_fine_tune_empty_valid_class_filter_returns_without_generation() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner,
        class_names=["walking"],
    )

    assert (
        wrapper.infer(
            frames=_frames(2),
            class_names=["not-a-model-class"],
            fps=5.0,
        )
        == []
    )
    assert reasoner.calls == []


def test_fine_tune_mode_detection_is_cached_at_construction() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner,
        class_names=["walking"],
    )
    tokenizer._added_vocabulary.clear()
    tokenizer._vocabulary.pop("<|cls:walking|>")

    wrapper.infer(frames=_frames(2), fps=5.0)

    assert reasoner.calls[0]["prefix_allowed_tokens_fn"] is not None


def test_constrained_decoder_allowed_tokens_follow_line_grammar() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    class_token_id = tokenizer.id("<|cls:walking|>")
    grammar = _FineTunePrefixAllowedTokensFn(
        tokenizer=tokenizer,
        class_token_ids={"walking": class_token_id},
    )

    start_allowed = grammar(0, torch.tensor([], dtype=torch.int64))
    assert class_token_id in start_allowed
    assert tokenizer.id("n") in start_allowed
    assert tokenizer.id("none") in start_allowed
    assert tokenizer.eos_token_id not in start_allowed

    prefix_through_open_angle = torch.tensor(
        [class_token_id, tokenizer.id(" "), tokenizer.id("<")]
    )
    after_open_angle = grammar(0, prefix_through_open_angle)
    assert tokenizer.id("0") in after_open_angle
    assert tokenizer.id("9") in after_open_angle
    assert tokenizer.id(".") not in after_open_angle
    assert tokenizer.eos_token_id not in after_open_angle

    complete_line = torch.tensor(
        [
            class_token_id,
            tokenizer.id(" "),
            tokenizer.id("<"),
            tokenizer.id("1"),
            tokenizer.id("."),
            tokenizer.id("2"),
            tokenizer.id(">"),
            tokenizer.id(" "),
            tokenizer.id("<"),
            tokenizer.id("3"),
            tokenizer.id("."),
            tokenizer.id("4"),
            tokenizer.id(">"),
        ]
    )
    line_boundary_allowed = grammar(0, complete_line)
    assert tokenizer.eos_token_id in line_boundary_allowed
    assert tokenizer.id("\n") in line_boundary_allowed
    assert class_token_id not in line_boundary_allowed

    after_newline = torch.cat(
        [complete_line, torch.tensor([tokenizer.id("\n")])]
    )
    assert grammar(0, after_newline) == [class_token_id]


def test_class_names_is_none_for_open_vocabulary_model() -> None:
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=_FakeReasoner())

    assert wrapper.class_names is None


def test_from_pretrained_wraps_loaded_reasoner(monkeypatch) -> None:
    reasoner = _FakeReasoner()
    calls = []

    def load_reasoner(model_name_or_path: str, **kwargs):
        calls.append((model_name_or_path, kwargs))
        return reasoner

    monkeypatch.setattr(Cosmos3EdgeReasoner, "from_pretrained", load_reasoner)

    wrapper = Cosmos3EdgeVideoSegmentClassification.from_pretrained(
        "nvidia/Cosmos3-Edge",
        local_files_only=False,
    )

    assert wrapper._reasoner is reasoner
    assert calls == [("nvidia/Cosmos3-Edge", {"local_files_only": False})]


def test_from_pretrained_reads_class_names_from_model_config(
    monkeypatch, tmp_path
) -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(tokenizer=tokenizer)
    (tmp_path / "model_config.json").write_text(
        json.dumps({"class_names": ["walking", "running"]})
    )
    monkeypatch.setattr(
        Cosmos3EdgeReasoner,
        "from_pretrained",
        MagicMock(return_value=reasoner),
    )

    wrapper = Cosmos3EdgeVideoSegmentClassification.from_pretrained(str(tmp_path))

    assert wrapper.class_names == ["walking", "running"]
    assert wrapper._fine_tune_prefix_allowed_tokens_fn is not None


def test_from_pretrained_derives_class_names_from_class_tokens(
    monkeypatch, tmp_path
) -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(tokenizer=tokenizer)
    monkeypatch.setattr(
        Cosmos3EdgeReasoner,
        "from_pretrained",
        MagicMock(return_value=reasoner),
    )

    wrapper = Cosmos3EdgeVideoSegmentClassification.from_pretrained(str(tmp_path))

    assert wrapper.class_names == ["walking", "running"]
    assert wrapper._fine_tune_prefix_allowed_tokens_fn is not None


def test_from_pretrained_reads_video_pre_processing(monkeypatch, tmp_path) -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    (tmp_path / "inference_config.json").write_text(
        json.dumps(
            {"video_pre_processing": {"max_frame_side": 100, "window_seconds": 8.0}}
        )
    )
    monkeypatch.setattr(
        Cosmos3EdgeReasoner,
        "from_pretrained",
        MagicMock(return_value=reasoner),
    )

    wrapper = Cosmos3EdgeVideoSegmentClassification.from_pretrained(str(tmp_path))
    assert wrapper.video_pre_processing == {
        "max_frame_side": 100,
        "window_seconds": 8.0,
    }

    large_frames = [np.zeros((480, 854, 3), dtype=np.uint8) for _ in range(4)]
    wrapper.infer(frames=large_frames, fps=2.0)
    sent_frames = reasoner.calls[0]["frames"]
    assert all(frame.shape == (56, 100, 3) for frame in sent_frames)


def test_fine_tune_parser_accepts_trailing_end_token(monkeypatch) -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(
        response=(
            "<|cls:walking|> <0.00> <1.00>\n"
            "<|cls:running|> <1.00> <2.00><|im_end|>"
        ),
        tokenizer=tokenizer,
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner, class_names=["walking", "running"]
    )

    result = wrapper.infer(frames=_frames(8), fps=4.0)

    assert [segment.class_name for segment in result] == ["walking", "running"]


def test_infer_accepts_chw_tensor_frames() -> None:
    reasoner = _FakeReasoner(response="moving")
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [torch.zeros((3, 8, 9), dtype=torch.uint8) for _ in range(3)]

    result = wrapper.infer(frames=frames, fps=4.0)

    processed_frames = reasoner.calls[0]["frames"]
    assert all(isinstance(frame, np.ndarray) for frame in processed_frames)
    assert all(frame.shape == (8, 9, 3) for frame in processed_frames)
    assert result == [_prediction(0, 2, "moving")]


def test_infer_requires_fps() -> None:
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=_FakeReasoner())

    with pytest.raises(ValueError, match="fps"):
        wrapper.infer(frames=_frames(1), class_names=["moving"])
