import json
from typing import Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.models.base.video_segment_classification import (
    VideoSegmentClassificationPrediction,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner
from inference_models.models.cosmos3.cosmos3_video_segment_classification import (
    Cosmos3EdgeVideoSegmentClassification,
    _parse_temporal_segments,
)


def _model_with_processor() -> Cosmos3EdgeReasoner:
    model = MagicMock()
    model.parameters.return_value = iter([torch.tensor(0.0, dtype=torch.bfloat16)])
    processor = MagicMock()
    processor.apply_chat_template.return_value = "templated"
    processor.return_value = {
        "input_ids": torch.tensor([[1, 2, 3]], dtype=torch.int64),
        "pixel_values": torch.zeros((1, 3, 8, 8), dtype=torch.float32),
    }
    return Cosmos3EdgeReasoner(
        model=model, processor=processor, device=torch.device("cpu")
    )


class _FakeReasoner:
    def __init__(self, response="[]", text_response="{}"):
        self.response = response
        self.text_response = text_response
        self.calls = []
        self.text_calls = []

    def prompt_video(self, **kwargs):
        self.calls.append(kwargs)
        return self.response

    def prompt_text(self, **kwargs):
        self.text_calls.append(kwargs)
        return self.text_response


def test_infer_defaults_to_the_temporal_localization_token_budget() -> None:
    reasoner = _FakeReasoner()
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8)]

    wrapper.infer(frames=frames, class_names=None, fps=5.0)
    wrapper.infer(frames=frames, class_names=None, fps=5.0, max_new_tokens=64)

    assert reasoner.calls[0]["max_new_tokens"] == 4096
    assert reasoner.calls[1]["max_new_tokens"] == 64


def test_infer_parses_segments_and_forwards_video_inputs() -> None:
    reasoner = _FakeReasoner(
        response={
            "answer": '[{"start": 0.2, "end": 0.6, "caption": "person walking by"}]'
        },
        text_response='{"1": "walking"}',
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]

    result = wrapper.infer(
        frames=frames,
        class_names=["walking", "running"],
        fps=5.0,
        max_new_tokens=128,
    )

    assert result == [
        VideoSegmentClassificationPrediction(
            start_frame_idx=1,
            end_frame_idx=3,
            class_name="walking",
        )
    ]
    assert len(reasoner.calls) == 1
    call = reasoner.calls[0]
    assert all(actual is expected for actual, expected in zip(call["frames"], frames))
    assert call["input_color_format"] == "rgb"
    assert call["video_fps"] == 5.0
    assert call["max_new_tokens"] == 128
    assert len(reasoner.text_calls) == 1
    mapping_prompt = reasoner.text_calls[0]["prompt"]
    assert "person walking by" in mapping_prompt
    assert '"walking"' in mapping_prompt


def test_infer_drops_segments_the_mapping_marks_as_other() -> None:
    reasoner = _FakeReasoner(
        response='[{"start": 0, "end": 0.2, "caption": "a robot idles"}, '
        '{"start": 0.2, "end": 0.4, "caption": "a person runs"}]',
        text_response='{"1": "other", "2": "running"}',
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(3)]

    result = wrapper.infer(frames=frames, class_names=["running"], fps=5.0)

    assert [segment.class_name for segment in result] == ["running"]
    assert result[0].start_frame_idx == 1


def test_infer_returns_empty_when_the_mapping_is_unparseable() -> None:
    reasoner = _FakeReasoner(
        response='[{"start": 0, "end": 0.2, "caption": "a person runs"}]',
        text_response="no json here",
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    result = wrapper.infer(frames=frames, class_names=["running"], fps=5.0)

    assert result == []


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
    reasoner = _FakeReasoner()
    (tmp_path / "model_config.json").write_text(
        json.dumps({"class_names": ["walking", "running"]})
    )
    monkeypatch.setattr(
        Cosmos3EdgeReasoner,
        "from_pretrained",
        MagicMock(return_value=reasoner),
    )

    wrapper = Cosmos3EdgeVideoSegmentClassification.from_pretrained(str(tmp_path))

    assert wrapper._reasoner is reasoner
    assert wrapper.class_names == ["walking", "running"]


def test_infer_parameter_vocabulary_overrides_model_vocabulary() -> None:
    reasoner = _FakeReasoner(
        response='[{"start": 0, "end": 0.2, "caption": "a person jumping"}]',
        text_response='{"1": "jumping"}',
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner,
        class_names=["walking"],
    )
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    result = wrapper.infer(frames=frames, class_names=["jumping"], fps=5.0)

    assert result == [
        VideoSegmentClassificationPrediction(
            start_frame_idx=0,
            end_frame_idx=1,
            class_name="jumping",
        )
    ]
    mapping_prompt = reasoner.text_calls[0]["prompt"]
    assert '"jumping"' in mapping_prompt
    assert "walking" not in mapping_prompt


def test_infer_uses_model_vocabulary_when_parameter_is_none() -> None:
    reasoner = _FakeReasoner(
        response='[{"start": 0, "end": 0.2, "caption": "a person walking by"}]',
        text_response='{"1": "walking"}',
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(
        reasoner=reasoner,
        class_names=["walking"],
    )
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    result = wrapper.infer(frames=frames, class_names=None, fps=5.0)

    assert [segment.class_name for segment in result] == ["walking"]
    assert 'action classes: ["walking"]' in reasoner.text_calls[0]["prompt"]


def test_infer_uses_open_vocabulary_prompt_and_condenses_captions() -> None:
    reasoner = _FakeReasoner(
        response='[{"start": 0, "end": 0.2, "caption": "a person opening a door"}]',
        text_response='{"1": "open door"}',
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    result = wrapper.infer(frames=frames, class_names=None, fps=5.0)

    assert result == [
        VideoSegmentClassificationPrediction(
            start_frame_idx=0,
            end_frame_idx=1,
            class_name="open door",
        )
    ]
    prompt = reasoner.calls[0]["prompt"]
    assert prompt.startswith("List all action segments in the video.")
    assert "'caption'" in prompt
    assert "Please list multiple events if applicable." in prompt
    assert '"caption": EVENT2' in prompt
    assert "Class vocabulary:" not in prompt
    label_prompt = reasoner.text_calls[0]["prompt"]
    assert "a person opening a door" in label_prompt
    assert "Condense each caption" in label_prompt


def test_open_vocabulary_labels_are_normalized_for_cross_call_merging() -> None:
    reasoner = _FakeReasoner(
        response='[{"start": 0, "end": 0.2, "caption": "the robot picks up a cup"}]',
        text_response='{"1": "Pick_Up  Green_Cup"}',
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    result = wrapper.infer(frames=frames, class_names=None, fps=5.0)

    assert [segment.class_name for segment in result] == ["pick up green cup"]


def test_open_vocabulary_condensing_falls_back_to_captions_when_unparseable() -> None:
    reasoner = _FakeReasoner(
        response='[{"start": 0, "end": 0.2, "caption": "a person opening a door"}]',
        text_response="no json here",
    )
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]

    result = wrapper.infer(frames=frames, class_names=None, fps=5.0)

    assert [segment.class_name for segment in result] == ["a person opening a door"]


@pytest.mark.parametrize(
    ("text", "class_names", "num_frames", "fps", "expected"),
    [
        (
            '[{"start": 0.21, "end": 1.01, "class": "running"}]',
            ["running"],
            10,
            5.0,
            [
                VideoSegmentClassificationPrediction(
                    start_frame_idx=1,
                    end_frame_idx=6,
                    class_name="running",
                )
            ],
        ),
        (
            # Integer seconds remain valid numeric timestamps.
            '[{"start": 0, "end": 1, "class": "running"}]',
            ["running"],
            10,
            5.0,
            [
                VideoSegmentClassificationPrediction(
                    start_frame_idx=0,
                    end_frame_idx=5,
                    class_name="running",
                )
            ],
        ),
        (
            # Numeric strings, booleans, missing labels, and non-string
            # labels are rejected.
            '[{"start": "0.1", "end": 0.3, "class": "jumping"}, '
            '{"start": true, "end": 0.3, "class": "jumping"}, '
            '{"start": 0.1, "end": 0.3}, '
            '{"start": 0.1, "end": 0.3, "class": 3}]',
            ["jumping"],
            8,
            4.0,
            [],
        ),
        (
            # The cookbook's open-vocabulary output labels events under a
            # "caption" key.
            '[{"start": 0.0, "end": 2.0, "caption": "robot reaches out"}, '
            '{"start": 2.0, "end": 4.0, "caption": "robot scoops popcorn"}]',
            None,
            32,
            4.0,
            [
                VideoSegmentClassificationPrediction(
                    start_frame_idx=0,
                    end_frame_idx=8,
                    class_name="robot reaches out",
                ),
                VideoSegmentClassificationPrediction(
                    start_frame_idx=8,
                    end_frame_idx=16,
                    class_name="robot scoops popcorn",
                ),
            ],
        ),
        (
            '[{"start": 0, "end": 1, "class": "unknown"}]',
            ["running"],
            8,
            4.0,
            [],
        ),
        (
            # Boundary timestamps clamp after conversion to frame indices.
            '[{"start": -0.3, "end": 0.4, "class": "running"}, '
            '{"start": 0.5, "end": 1.4, "class": "running"}]',
            ["running"],
            5,
            5.0,
            [
                VideoSegmentClassificationPrediction(
                    start_frame_idx=0,
                    end_frame_idx=2,
                    class_name="running",
                ),
                VideoSegmentClassificationPrediction(
                    start_frame_idx=2,
                    end_frame_idx=4,
                    class_name="running",
                ),
            ],
        ),
        (
            # Intervals wholly before or after the clip are absurd and dropped.
            '[{"start": -0.4, "end": -0.1, "class": "running"}, '
            '{"start": 2.1, "end": 2.4, "class": "running"}]',
            ["running"],
            10,
            5.0,
            [],
        ),
        (
            '[{"start": 1.4, "end": 0.2, "class": "running"}]',
            ["running"],
            10,
            5.0,
            [
                VideoSegmentClassificationPrediction(
                    start_frame_idx=1,
                    end_frame_idx=7,
                    class_name="running",
                )
            ],
        ),
        (
            '[{"start": 0, "end": 0.2, "class": "walking"}] '
            '[{"start": 0.2, "end": 0.4, "class": "running"}]',
            None,
            10,
            5.0,
            [
                VideoSegmentClassificationPrediction(
                    start_frame_idx=0,
                    end_frame_idx=1,
                    class_name="walking",
                )
            ],
        ),
        ("The video contains somebody walking.", ["walking"], 10, 5.0, []),
        (
            'analysis about the clip</think> '
            '[{"start": 0, "end": 0.4, "class": "walking"}]',
            ["walking"],
            10,
            5.0,
            [
                VideoSegmentClassificationPrediction(
                    start_frame_idx=0,
                    end_frame_idx=2,
                    class_name="walking",
                )
            ],
        ),
        (
            'Result:\n```json\n[{"start": 0.2, "end": 0.6, '
            '"class": "walking"}]\n```',
            ["walking"],
            10,
            5.0,
            [
                VideoSegmentClassificationPrediction(
                    start_frame_idx=1,
                    end_frame_idx=3,
                    class_name="walking",
                )
            ],
        ),
        (
            '[{"start": "soon", "end": 0.2, "class": "walking"}, '
            '{"start": 0.2, "end": 0.6, "class": "walking"}]',
            ["walking"],
            10,
            5.0,
            [
                VideoSegmentClassificationPrediction(
                    start_frame_idx=1,
                    end_frame_idx=3,
                    class_name="walking",
                )
            ],
        ),
    ],
)
def test_parse_temporal_segments(
    text: str,
    class_names: Optional[list[str]],
    num_frames: int,
    fps: float,
    expected: list[VideoSegmentClassificationPrediction],
) -> None:
    assert _parse_temporal_segments(text, class_names, num_frames, fps) == expected


def test_infer_sends_the_cookbook_prompt_through_the_reasoner() -> None:
    reasoner = _model_with_processor()
    reasoner._model.generate.return_value = torch.tensor([[1, 2, 3, 9]])
    reasoner._processor.batch_decode.side_effect = [
        ['[{"start": 0, "end": 0.4, "caption": "a person walking by"}]'],
        ['{"1": "walking"}'],
    ]
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]

    result = wrapper.infer(frames=frames, class_names=None, fps=5.0)

    conversation = reasoner._processor.apply_chat_template.call_args_list[0].args[0]
    prompt = conversation[1]["content"][1]["text"]
    assert prompt.startswith("List all action segments in the video.")
    assert "'caption'" in prompt
    assert "Please list multiple events if applicable." in prompt
    label_conversation = reasoner._processor.apply_chat_template.call_args_list[
        1
    ].args[0]
    assert "Condense each caption" in label_conversation[1]["content"][0]["text"]
    assert result == [
        VideoSegmentClassificationPrediction(
            start_frame_idx=0,
            end_frame_idx=2,
            class_name="walking",
        )
    ]


def test_infer_accepts_chw_tensor_frames() -> None:
    reasoner = _model_with_processor()
    reasoner._model.generate.return_value = torch.tensor([[1, 2, 3, 9]])
    reasoner._processor.batch_decode.side_effect = [
        ['[{"start": 0, "end": 0.5, "caption": "something moving"}]'],
        ['{"1": "moving"}'],
    ]
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=reasoner)
    frames = [torch.zeros((3, 8, 9), dtype=torch.uint8) for _ in range(3)]

    result = wrapper.infer(
        frames=frames,
        class_names=None,
        fps=4.0,
    )

    processed_frames = reasoner._processor.call_args_list[0].kwargs["videos"][0]
    assert all(isinstance(frame, np.ndarray) for frame in processed_frames)
    assert all(frame.shape == (8, 9, 3) for frame in processed_frames)
    assert result == [
        VideoSegmentClassificationPrediction(
            start_frame_idx=0,
            end_frame_idx=2,
            class_name="moving",
        )
    ]


def test_infer_requires_fps() -> None:
    wrapper = Cosmos3EdgeVideoSegmentClassification(reasoner=_model_with_processor())

    with pytest.raises(ValueError, match="fps"):
        wrapper.infer(
            frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
            class_names=["moving"],
        )
