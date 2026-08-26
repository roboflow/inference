import numpy as np

from inference_models.models.base.video_classification import (
    VideoSegmentClassification,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner
from inference_models.models.cosmos3.cosmos3_video_classification import (
    Cosmos3VideoSegmentClassification,
)


class _FakeReasoner:
    def __init__(self):
        self.calls = []

    def temporal_localization(self, **kwargs):
        self.calls.append(kwargs)
        return [
            {
                "start_frame_idx": 1,
                "end_frame_idx": 3,
                "class": "walking",
            }
        ]


def test_infer_converts_segments_and_forwards_video_inputs() -> None:
    reasoner = _FakeReasoner()
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
    assert reasoner.calls == [
        {
            "frames": frames,
            "class_names": ["walking", "running"],
            "input_color_format": "rgb",
            "fps": 5.0,
            "max_new_tokens": 128,
        }
    ]


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
