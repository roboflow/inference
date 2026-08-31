import json
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.base.action_recognition import (
    SLIDING_WINDOW_MODE,
    WHOLE_VIDEO_MODE,
    ActionRecognitionPrediction,
)
from inference_models.models.cosmos3 import cosmos3_action_recognition
from inference_models.models.cosmos3.cosmos3_action_recognition import (
    FINE_TUNE_MAX_NEW_TOKENS,
    FINE_TUNE_SYSTEM_PROMPT,
    ZERO_SHOT_MAX_NEW_TOKENS,
    ZERO_SHOT_TEMPORAL_LOCALIZATION_PROMPT,
    Cosmos3EdgeActionRecognition,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import (
    SYSTEM_PROMPT_SENTINEL,
    Cosmos3EdgeReasoner,
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
    return ActionRecognitionPrediction(
        start_frame_idx=start,
        end_frame_idx=end,
        class_name=label,
    )


def _cookbook_response(entries=None) -> str:
    """The cookbook answer shape: a JSON list wrapped in the model's prose."""
    entries = (
        entries
        if entries is not None
        else [{"start": 0.0, "end": 2.0, "caption": "a person walks"}]
    )
    return (
        "Looking at the clip, I can see the following events.\n\n"
        "```json\n" + json.dumps(entries, indent=1) + "\n```"
    )


def test_class_token_presence_routes_to_fine_tune_mode_for_long_clip() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    wrapper = Cosmos3EdgeActionRecognition(
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
    reasoner = _FakeReasoner(response=_cookbook_response(), tokenizer=_FakeTokenizer())
    wrapper = Cosmos3EdgeActionRecognition(
        reasoner=reasoner,
        class_names=["walking", "running"],
    )

    result = wrapper.infer(frames=_frames(200), fps=2.0)

    assert result == [_prediction(0, 4, "a person walks")]
    call = reasoner.calls[0]
    assert call["max_new_tokens"] == ZERO_SHOT_MAX_NEW_TOKENS
    assert call["prefix_allowed_tokens_fn"] is None
    assert SYSTEM_PROMPT_SENTINEL not in call["prompt"]
    assert call["prompt"] == ZERO_SHOT_TEMPORAL_LOCALIZATION_PROMPT
    # The reasoning pass is what keeps the output dense.
    assert call["enable_thinking"] is True


def test_fine_tune_mode_caps_frame_side_at_the_training_resolution() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    wrapper = Cosmos3EdgeActionRecognition(reasoner=reasoner, class_names=["walking"])

    large_frames = [np.zeros((480, 854, 3), dtype=np.uint8) for _ in range(4)]
    wrapper.infer(frames=large_frames, fps=2.0)

    sent_frames = reasoner.calls[0]["frames"]
    assert all(frame.shape == (202, 360, 3) for frame in sent_frames)


def test_zero_shot_mode_keeps_native_frame_resolution() -> None:
    reasoner = _FakeReasoner(response=_cookbook_response())
    wrapper = Cosmos3EdgeActionRecognition(reasoner=reasoner)

    large_frames = [np.zeros((480, 854, 3), dtype=np.uint8) for _ in range(4)]
    wrapper.infer(frames=large_frames, class_names=["walking"], fps=2.0)

    sent_frames = reasoner.calls[0]["frames"]
    assert all(frame.shape == (480, 854, 3) for frame in sent_frames)


def test_from_pretrained_rejects_class_names_that_miss_their_tokens(
    monkeypatch, tmp_path
) -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    reasoner = _FakeReasoner(tokenizer=tokenizer)
    # The package lists a class the tokenizer has no token for.
    (tmp_path / "class_names.txt").write_text("walking\nrunning\n")
    monkeypatch.setattr(
        Cosmos3EdgeReasoner, "from_pretrained", MagicMock(return_value=reasoner)
    )

    with pytest.raises(CorruptedModelPackageError):
        Cosmos3EdgeActionRecognition.from_pretrained(str(tmp_path))


def test_zero_shot_uses_caller_max_new_tokens() -> None:
    reasoner = _FakeReasoner(response=_cookbook_response())
    wrapper = Cosmos3EdgeActionRecognition(reasoner=reasoner)

    wrapper.infer(
        frames=_frames(2),
        class_names=["walking"],
        fps=5.0,
        max_new_tokens=17,
    )

    assert reasoner.calls[0]["max_new_tokens"] == 17


def test_fine_tune_prompt_contains_exact_legend_instruction_and_system_prompt() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    wrapper = Cosmos3EdgeActionRecognition(
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


def test_fine_tune_class_filter_drops_non_members_and_limits_parser() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(
        response=("<|cls:walking|> <0.00> <0.20>\n" "<|cls:running|> <0.20> <0.40>"),
        tokenizer=tokenizer,
    )
    wrapper = Cosmos3EdgeActionRecognition(
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
    wrapper = Cosmos3EdgeActionRecognition(
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
    wrapper = Cosmos3EdgeActionRecognition(
        reasoner=reasoner,
        class_names=["walking"],
    )
    tokenizer._added_vocabulary.clear()
    tokenizer._vocabulary.pop("<|cls:walking|>")

    wrapper.infer(frames=_frames(2), fps=5.0)

    assert reasoner.calls[0]["prefix_allowed_tokens_fn"] is not None


def test_class_names_is_none_for_open_vocabulary_model() -> None:
    wrapper = Cosmos3EdgeActionRecognition(reasoner=_FakeReasoner())

    assert wrapper.class_names is None


def test_from_pretrained_wraps_loaded_reasoner(monkeypatch) -> None:
    reasoner = _FakeReasoner()
    calls = []

    def load_reasoner(model_name_or_path: str, **kwargs):
        calls.append((model_name_or_path, kwargs))
        return reasoner

    monkeypatch.setattr(Cosmos3EdgeReasoner, "from_pretrained", load_reasoner)

    wrapper = Cosmos3EdgeActionRecognition.from_pretrained(
        "nvidia/Cosmos3-Edge",
        local_files_only=False,
    )

    assert wrapper._reasoner is reasoner
    assert calls == [("nvidia/Cosmos3-Edge", {"local_files_only": False})]


def test_from_pretrained_reads_class_names_file(monkeypatch, tmp_path) -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(tokenizer=tokenizer)
    (tmp_path / "class_names.txt").write_text("walking\nrunning\n")
    monkeypatch.setattr(
        Cosmos3EdgeReasoner,
        "from_pretrained",
        MagicMock(return_value=reasoner),
    )

    wrapper = Cosmos3EdgeActionRecognition.from_pretrained(str(tmp_path))

    assert wrapper.class_names == ["walking", "running"]
    assert wrapper._fine_tune_prefix_allowed_tokens_fn is not None


def test_from_pretrained_rejects_a_fine_tune_without_class_names(
    monkeypatch, tmp_path
) -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(tokenizer=tokenizer)
    monkeypatch.setattr(
        Cosmos3EdgeReasoner,
        "from_pretrained",
        MagicMock(return_value=reasoner),
    )

    with pytest.raises(CorruptedModelPackageError):
        Cosmos3EdgeActionRecognition.from_pretrained(str(tmp_path))


def test_from_pretrained_accepts_a_base_model_without_class_names(
    monkeypatch, tmp_path
) -> None:
    reasoner = _FakeReasoner(tokenizer=_FakeTokenizer())
    monkeypatch.setattr(
        Cosmos3EdgeReasoner,
        "from_pretrained",
        MagicMock(return_value=reasoner),
    )

    wrapper = Cosmos3EdgeActionRecognition.from_pretrained(str(tmp_path))

    assert wrapper.class_names is None
    assert wrapper._fine_tune_prefix_allowed_tokens_fn is None


def test_from_pretrained_reads_the_recorded_sampling(monkeypatch, tmp_path) -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    reasoner = _FakeReasoner(response="none", tokenizer=tokenizer)
    (tmp_path / "class_names.txt").write_text("walking\n")
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

    wrapper = Cosmos3EdgeActionRecognition.from_pretrained(str(tmp_path))
    assert wrapper.video_sampling.max_frame_side == 100
    assert wrapper.video_sampling.window_seconds == 8.0

    large_frames = [np.zeros((480, 854, 3), dtype=np.uint8) for _ in range(4)]
    wrapper.infer(frames=large_frames, fps=2.0)
    sent_frames = reasoner.calls[0]["frames"]
    assert all(frame.shape == (56, 100, 3) for frame in sent_frames)


def test_fine_tune_parser_accepts_trailing_end_token(monkeypatch) -> None:
    tokenizer = _FakeTokenizer(class_names=["walking", "running"])
    reasoner = _FakeReasoner(
        response=(
            "<|cls:walking|> <0.00> <1.00>\n" "<|cls:running|> <1.00> <2.00><|im_end|>"
        ),
        tokenizer=tokenizer,
    )
    wrapper = Cosmos3EdgeActionRecognition(
        reasoner=reasoner, class_names=["walking", "running"]
    )

    result = wrapper.infer(frames=_frames(8), fps=4.0)

    assert [segment.class_name for segment in result] == ["walking", "running"]


def test_infer_accepts_chw_tensor_frames() -> None:
    reasoner = _FakeReasoner(
        response=_cookbook_response([{"start": 0.0, "end": 0.5, "caption": "moving"}])
    )
    wrapper = Cosmos3EdgeActionRecognition(reasoner=reasoner)
    frames = [torch.zeros((3, 8, 9), dtype=torch.uint8) for _ in range(3)]

    result = wrapper.infer(frames=frames, fps=4.0)

    processed_frames = reasoner.calls[0]["frames"]
    assert all(isinstance(frame, np.ndarray) for frame in processed_frames)
    assert all(frame.shape == (8, 9, 3) for frame in processed_frames)
    assert result == [_prediction(0, 2, "moving")]


def test_infer_requires_fps() -> None:
    wrapper = Cosmos3EdgeActionRecognition(reasoner=_FakeReasoner())

    with pytest.raises(ValueError, match="fps"):
        wrapper.infer(frames=_frames(1), class_names=["moving"])


def test_zero_shot_reads_every_event_the_cookbook_list_holds() -> None:
    reasoner = _FakeReasoner(
        response=_cookbook_response(
            [
                {"start": 0.0, "end": 2.0, "caption": "a person walks"},
                {"start": 3.0, "end": 5.0, "caption": "a person sits down"},
            ]
        )
    )
    wrapper = Cosmos3EdgeActionRecognition(reasoner=reasoner)

    result = wrapper.infer(frames=_frames(40), fps=4.0)

    # Seconds become frame indices at the rate the frames were drawn at.
    assert result == [
        _prediction(0, 8, "a person walks"),
        _prediction(12, 20, "a person sits down"),
    ]


def test_zero_shot_clamps_segments_to_the_frames_it_was_given() -> None:
    reasoner = _FakeReasoner(
        response=_cookbook_response(
            [{"start": -1.0, "end": 999.0, "caption": "the whole clip"}]
        )
    )
    wrapper = Cosmos3EdgeActionRecognition(reasoner=reasoner)

    result = wrapper.infer(frames=_frames(10), fps=4.0)

    assert result == [_prediction(0, 9, "the whole clip")]


@pytest.mark.parametrize(
    "response",
    [
        "",
        "no json here at all",
        '```json\n{"start": 0}\n```',
        '```json\n[{"start": 0.0, "end": 1.0}]\n```',
        '```json\n[{"caption": "walking"}]\n```',
        '```json\n[{"start": null, "end": 1.0, "caption": "walking"}]\n```',
    ],
)
def test_zero_shot_returns_nothing_when_the_answer_does_not_parse(response) -> None:
    reasoner = _FakeReasoner(response=response)
    wrapper = Cosmos3EdgeActionRecognition(reasoner=reasoner)

    assert wrapper.infer(frames=_frames(10), fps=4.0) == []


def test_zero_shot_ignores_a_requested_vocabulary_and_says_so(monkeypatch) -> None:
    # The checkpoint ignores classes stated inside the localization prompt,
    # so the wrapper says so rather than look like it applied them.
    reasoner = _FakeReasoner(response=_cookbook_response())
    wrapper = Cosmos3EdgeActionRecognition(reasoner=reasoner)
    warn = MagicMock()
    monkeypatch.setattr(cosmos3_action_recognition.LOGGER, "warning", warn)

    result = wrapper.infer(
        frames=_frames(10), class_names=["walking", "running"], fps=4.0
    )

    assert result == [_prediction(0, 8, "a person walks")]
    warn.assert_called_once()
    assert "ignored" in warn.call_args.args[0]
    assert "walking" not in reasoner.calls[0]["prompt"]


def test_zero_shot_declares_whole_video_sampling() -> None:
    # Zero-shot has no trained window, so it reads a clip in one call.
    wrapper = Cosmos3EdgeActionRecognition(reasoner=_FakeReasoner(response="none"))

    assert wrapper.video_sampling.mode == WHOLE_VIDEO_MODE


def test_a_fine_tune_keeps_the_sampling_mode_its_package_declares() -> None:
    tokenizer = _FakeTokenizer(class_names=["walking"])
    wrapper = Cosmos3EdgeActionRecognition(
        reasoner=_FakeReasoner(response="none", tokenizer=tokenizer),
        class_names=["walking"],
    )

    assert wrapper.video_sampling.mode == SLIDING_WINDOW_MODE
