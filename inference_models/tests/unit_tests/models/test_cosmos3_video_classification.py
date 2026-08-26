from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from inference_models.models.base.video_segment_classification import (
    VideoSegmentClassification,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner
from inference_models.models.cosmos3.cosmos3_video_classification import (
    Cosmos3VideoSegmentClassification,
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
    def __init__(self, response="[]"):
        self.response = response
        self.calls = []

    def prompt_video(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def test_infer_parses_segments_and_forwards_video_inputs() -> None:
    reasoner = _FakeReasoner(
        response={
            "answer": '[{"start": 1, "end": 3, "class": "walking"}]'
        }
    )
    wrapper = Cosmos3VideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]

    result = wrapper.infer(
        frames=frames,
        class_names=["walking", "running"],
        fps=5.0,
        max_new_tokens=128,
    )

    assert result == [
        VideoSegmentClassification(
            start_frame_idx=1,
            end_frame_idx=3,
            class_name="walking",
        )
    ]
    assert len(reasoner.calls) == 1
    call = reasoner.calls[0]
    assert all(actual is expected for actual, expected in zip(call["frames"], frames))
    assert call["input_color_format"] == "rgb"
    assert call["max_new_tokens"] == 128


def test_class_names_is_none_for_open_vocabulary_model() -> None:
    wrapper = Cosmos3VideoSegmentClassification(reasoner=_FakeReasoner())

    assert wrapper.class_names is None


def test_from_pretrained_wraps_loaded_reasoner(monkeypatch) -> None:
    reasoner = _FakeReasoner()
    calls = []

    def load_reasoner(model_name_or_path: str, **kwargs):
        calls.append((model_name_or_path, kwargs))
        return reasoner

    monkeypatch.setattr(Cosmos3EdgeReasoner, "from_pretrained", load_reasoner)

    wrapper = Cosmos3VideoSegmentClassification.from_pretrained(
        "nvidia/Cosmos3-Edge",
        local_files_only=False,
    )

    assert wrapper._reasoner is reasoner
    assert calls == [("nvidia/Cosmos3-Edge", {"local_files_only": False})]


@pytest.mark.parametrize(
    ("text", "class_names", "num_frames", "expected"),
    [
        (
            '[{"start": 2, "end": 5, "class": "running"}]',
            ["running"],
            10,
            [
                VideoSegmentClassification(
                    start_frame_idx=2,
                    end_frame_idx=5,
                    class_name="running",
                )
            ],
        ),
        (
            # Only plain JSON integers under "class" survive: floats,
            # numeric strings, and the "caption" key are all rejected.
            '[{"start": 1.0, "end": "3", "caption": "jumping"}]',
            ["jumping"],
            8,
            [],
        ),
        (
            # Non-integer indices mean the model ignored the schema: drop.
            '[{"start": 0.21, "end": 1.01, "class": "running"}]',
            ["running"],
            10,
            [],
        ),
        (
            # Timestamps are not converted: drop.
            '[{"start": "00:01.25", "end": "00:02.01", '
            '"class": "running"}]',
            ["running"],
            10,
            [],
        ),
        (
            '[{"start": 0, "end": 1, "class": "unknown"}]',
            ["running"],
            8,
            [],
        ),
        (
            # Out-of-range indices are invalid: drop, no clamping.
            '[{"start": -3, "end": 2, "class": "running"}, '
            '{"start": 0, "end": 100, "class": "running"}]',
            ["running"],
            5,
            [],
        ),
        (
            '[{"start": 2, "end": 1, "class": "running"}]',
            ["running"],
            10,
            [
                VideoSegmentClassification(
                    start_frame_idx=1,
                    end_frame_idx=2,
                    class_name="running",
                )
            ],
        ),
        ("The video contains somebody walking.", ["walking"], 10, []),
        (
            'analysis about the clip</think> '
            '[{"start": 0, "end": 2, "class": "walking"}]',
            ["walking"],
            10,
            [
                VideoSegmentClassification(
                    start_frame_idx=0,
                    end_frame_idx=2,
                    class_name="walking",
                )
            ],
        ),
        (
            'Result:\n```json\n[{"start": 1, "end": 3, '
            '"class": "walking"}]\n```',
            ["walking"],
            10,
            [
                VideoSegmentClassification(
                    start_frame_idx=1,
                    end_frame_idx=3,
                    class_name="walking",
                )
            ],
        ),
        (
            '[{"start": "soon", "end": 1, "class": "walking"}, '
            '{"start": 1, "end": 3, "class": "walking"}]',
            ["walking"],
            10,
            [
                VideoSegmentClassification(
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
    class_names: list[str],
    num_frames: int,
    expected: list[VideoSegmentClassification],
) -> None:
    assert _parse_temporal_segments(text, class_names, num_frames) == expected


def test_infer_builds_prompt_with_clip_metadata() -> None:
    reasoner = _model_with_processor()
    reasoner._model.generate.return_value = torch.tensor([[1, 2, 3, 9]])
    reasoner._processor.batch_decode.return_value = [
        '[{"start": 0, "end": 2, "class": "walking"}]'
    ]
    wrapper = Cosmos3VideoSegmentClassification(reasoner=reasoner)
    frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(4)]

    result = wrapper.infer(
        frames=frames,
        class_names=["walking", "jumping"],
        fps=5.0,
    )

    conversation = reasoner._processor.apply_chat_template.call_args.args[0]
    prompt = conversation[1]["content"][1]["text"]
    assert "walking" in prompt
    assert "jumping" in prompt
    assert "4 frames" in prompt
    assert "5.0 fps" in prompt
    assert "frame indices between 0 and 3" in prompt
    assert result == [
        VideoSegmentClassification(
            start_frame_idx=0,
            end_frame_idx=2,
            class_name="walking",
        )
    ]


def test_infer_accepts_chw_tensor_frames() -> None:
    reasoner = _model_with_processor()
    reasoner._model.generate.return_value = torch.tensor([[1, 2, 3, 9]])
    reasoner._processor.batch_decode.return_value = [
        '[{"start": 0, "end": 2, "class": "moving"}]'
    ]
    wrapper = Cosmos3VideoSegmentClassification(reasoner=reasoner)
    frames = [torch.zeros((3, 8, 9), dtype=torch.uint8) for _ in range(3)]

    result = wrapper.infer(
        frames=frames,
        class_names=["moving"],
        fps=4.0,
    )

    processed_frames = reasoner._processor.call_args.kwargs["videos"][0]
    assert all(isinstance(frame, np.ndarray) for frame in processed_frames)
    assert all(frame.shape == (8, 9, 3) for frame in processed_frames)
    assert result == [
        VideoSegmentClassification(
            start_frame_idx=0,
            end_frame_idx=2,
            class_name="moving",
        )
    ]


def test_infer_requires_fps() -> None:
    wrapper = Cosmos3VideoSegmentClassification(reasoner=_model_with_processor())

    with pytest.raises(ValueError, match="fps"):
        wrapper.infer(
            frames=[np.zeros((8, 8, 3), dtype=np.uint8)],
            class_names=["moving"],
        )
