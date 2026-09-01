import contextlib
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from inference.core.entities.requests.action_recognition import (
    ActionRecognitionInferenceRequest,
    InferenceRequestVideo,
)
from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction,
    VideoSampling,
)

MODULE = "inference.core.models.inference_models_adapters"


class _FakeModel:
    def __init__(self, responses, class_names=None, sampling=None):
        self.responses = list(responses)
        self._class_names = class_names
        self._sampling = sampling or VideoSampling()
        self.calls = []

    @property
    def class_names(self):
        return self._class_names

    @property
    def video_sampling(self):
        return self._sampling

    def infer(self, frames, class_names=None, fps=None, **kwargs):
        self.calls.append(
            {"frames": len(frames), "class_names": class_names, "fps": fps}
        )
        return self.responses.pop(0) if self.responses else []


def _adapter(model):
    from inference.core.models.inference_models_adapters import (
        InferenceModelsActionRecognitionAdapter,
    )

    adapter = InferenceModelsActionRecognitionAdapter.__new__(
        InferenceModelsActionRecognitionAdapter
    )
    adapter._model = model
    return adapter


def _request(class_filter=None):
    return ActionRecognitionInferenceRequest(
        model_id="workspace/model",
        video=InferenceRequestVideo(type="base64", value="Zm9v"),
        class_filter=class_filter,
    )


@contextlib.contextmanager
def _clip(frame_count: int, source_fps: float):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with patch(f"{MODULE}.video_source_path") as source_path, patch(
        f"{MODULE}.probe_video", return_value=(source_fps, frame_count)
    ), patch(
        f"{MODULE}.read_frame_windows",
        side_effect=lambda path, windows, max_frame_side: (
            [frame] * len(window) for window in windows
        ),
    ):
        source_path.return_value.__enter__ = MagicMock(return_value="/tmp/clip")
        source_path.return_value.__exit__ = MagicMock(return_value=False)
        yield


def test_window_segments_map_to_clip_frame_indices() -> None:
    model = _FakeModel(
        responses=[[ActionRecognitionPrediction(0, 15, "walk")]],
        class_names=["walk", "run"],
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0, min_frames=4),
    )
    with _clip(frame_count=100, source_fps=10.0):
        response = _adapter(model).infer_from_request(_request())

    assert response.windows_classified == 1
    assert response.source_fps == 10.0
    assert response.frame_count == 100
    # The model reported its own frames 0..15; the clip counts 0..75.
    assert response.timeline[0].start_frame_idx == 0
    assert response.timeline[0].end_frame_idx == 75
    assert response.timeline[0].class_name == "walk"
    assert response.timeline[0].class_id == 0


def test_ranges_of_one_class_merge_across_windows() -> None:
    model = _FakeModel(
        responses=[
            [ActionRecognitionPrediction(0, 15, "walk")],
            [ActionRecognitionPrediction(0, 15, "walk")],
        ],
        class_names=["walk"],
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0, min_frames=4),
    )
    with _clip(frame_count=170, source_fps=10.0):
        response = _adapter(model).infer_from_request(_request())

    assert response.windows_classified == 2
    # Neighbouring windows of one class come back as a single range.
    assert len(response.timeline) == 1
    assert response.timeline[0].start_frame_idx == 0
    assert response.timeline[0].end_frame_idx == 155


def test_the_model_is_never_told_more_than_the_source_frame_rate() -> None:
    model = _FakeModel(
        responses=[[]],
        class_names=["walk"],
        sampling=VideoSampling(window_seconds=8.0, sample_fps=30.0, min_frames=4),
    )
    with _clip(frame_count=100, source_fps=10.0):
        _adapter(model).infer_from_request(_request())

    assert model.calls[0]["fps"] == 10.0


def test_class_filter_reaches_the_model() -> None:
    model = _FakeModel(responses=[[]], class_names=["walk", "run"])
    with _clip(frame_count=1000, source_fps=10.0):
        _adapter(model).infer_from_request(_request(class_filter=["run"]))

    assert model.calls[0]["class_names"] == ["run"]


def test_a_very_short_clip_is_still_classified() -> None:
    # Training clamps the sample count up to its floor rather than refusing,
    # so a clip it trained on stays servable here.
    model = _FakeModel(responses=[[]], class_names=["walk"])
    with _clip(frame_count=3, source_fps=10.0):
        response = _adapter(model).infer_from_request(_request())

    assert response.timeline == []
    assert response.windows_classified == 1
    assert len(model.calls) == 1


def test_windows_classified_counts_calls_not_plans() -> None:
    # A truncated container plans a window whose frames never decode. The
    # field names model calls, so a skipped window must not be counted.
    model = _FakeModel(responses=[[]], class_names=["walk"])
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with patch(f"{MODULE}.video_source_path") as source_path, patch(
        f"{MODULE}.probe_video", return_value=(10.0, 80)
    ), patch(
        f"{MODULE}.read_frame_windows",
        side_effect=lambda path, windows, max_frame_side: ([frame] for _ in windows),
    ):
        source_path.return_value.__enter__ = MagicMock(return_value="/tmp/clip")
        source_path.return_value.__exit__ = MagicMock(return_value=False)
        response = _adapter(model).infer_from_request(_request())

    assert response.windows_classified == 0
    assert model.calls == []


def test_an_open_vocabulary_label_reports_no_class_id() -> None:
    model = _FakeModel(
        responses=[[ActionRecognitionPrediction(0, 15, "cars on a road")]],
        class_names=None,
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0, min_frames=4),
    )
    with _clip(frame_count=100, source_fps=10.0):
        response = _adapter(model).infer_from_request(_request())

    assert response.timeline[0].class_id == -1


def test_the_wire_shape_names_the_class_field_class() -> None:
    model = _FakeModel(
        responses=[[ActionRecognitionPrediction(0, 15, "walk")]],
        class_names=["walk"],
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0, min_frames=4),
    )
    with _clip(frame_count=100, source_fps=10.0):
        response = _adapter(model).infer_from_request(_request())

    serialized = response.model_dump(by_alias=True)
    assert "class" in serialized["timeline"][0]
    assert "class_name" not in serialized["timeline"][0]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("walk,run", ["walk", "run"]),
        (" walk , run ", ["walk", "run"]),
        ("walk", ["walk"]),
        ("", None),
        (None, None),
        (",, ,", None),
    ],
)
def test_the_legacy_route_reads_a_comma_separated_class_list(raw, expected) -> None:
    from inference.core.interfaces.http.http_api import _parse_legacy_class_filter

    assert _parse_legacy_class_filter(class_filter=raw) == expected


def test_a_request_filter_does_not_become_a_class_vocabulary() -> None:
    # A zero-shot model ignores the filter and answers in its own words. A
    # caption matching one of the requested names must not inherit its index.
    model = _FakeModel(
        responses=[[ActionRecognitionPrediction(0, 15, "running")]],
        class_names=None,
    )
    with _clip(frame_count=64, source_fps=4.0):
        response = _adapter(model).infer_from_request(
            _request(class_filter=["walking", "running"])
        )

    assert [entry.class_name for entry in response.timeline] == ["running"]
    assert [entry.class_id for entry in response.timeline] == [-1]
