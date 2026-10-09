import contextlib
import importlib
from types import SimpleNamespace
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

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
    resolved_model: Optional[SimpleNamespace] = None
    supports_confidence = False
    supports_observed_duration = False

    def estimate_candidate_count(self, sampled_frames, *, class_names=None):
        return None

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


def test_unscored_adapter_loads_without_candidate_confidence_helper(monkeypatch):
    action_recognition = importlib.import_module(
        "inference_models.models.base.action_recognition"
    )
    adapters = importlib.import_module(MODULE)

    with monkeypatch.context() as package:
        package.delattr(action_recognition, "action_confidence_mask", raising=False)
        importlib.reload(adapters)

        model = _FakeModel(
            responses=[[ActionRecognitionPrediction(0, 1, "walk")]],
            class_names=["walk"],
        )
        request = _request()
        request.confidence = 0.9
        request.include_candidates = True
        with _clip(frame_count=3, source_fps=10.0):
            response = _adapter(model).infer_from_request(request)

    assert len(response.timeline) == 1
    assert response.candidates is None
    assert response.confidence_threshold is None
    assert len(model.calls) == 1


@pytest.mark.parametrize("confidence", [0.9, "best"])
def test_candidates_preserve_timeline_at_float32_threshold_boundaries(confidence):
    class ScoredModel(_FakeModel):
        supports_confidence = True
        confidence_threshold = 0.9
        recommended_parameters = SimpleNamespace(
            confidence=0.9, per_class_confidence={"walk": 0.9, "run": 0.9}
        )

        def infer(self, frames, class_names=None, fps=None, **kwargs):
            self.calls.append(kwargs)
            scores = torch.tensor([0.9, 0.9 - 1e-6], dtype=torch.float32)
            threshold = 0.0 if kwargs.get("confidence") == 0.0 else 0.9
            return [
                ActionRecognitionPrediction(0, 1, label, score.item(), True)
                for label, score in zip(self.class_names, scores)
                if score >= threshold
            ]

    responses = []
    for include_candidates in (False, True):
        model = ScoredModel([], class_names=["walk", "run"])
        request = _request()
        request.confidence = confidence
        request.include_candidates = include_candidates
        with _clip(frame_count=3, source_fps=10.0):
            responses.append(_adapter(model).infer_from_request(request))
        assert len(model.calls) == 1

    assert responses[0].timeline == responses[1].timeline
    assert len(responses[0].timeline) == 1
    assert responses[0].timeline[0].class_name == "walk"
    assert len(responses[1].candidates) == 2


@pytest.mark.parametrize("known_estimate", [False, True])
def test_candidate_overflow_fails_before_response_or_known_oversized_inference(
    monkeypatch, known_estimate
):
    from inference.core.exceptions import PayloadTooLargeError
    from inference.core.models import inference_models_adapters as adapters

    model = _FakeModel(
        responses=[[ActionRecognitionPrediction(0, 1, "walk", 0.9, True)] * 2],
        class_names=["walk"],
    )
    model.supports_confidence = True
    model.confidence_threshold = 0.5
    model.estimate_candidate_count = lambda *args, **kwargs: (
        2 if known_estimate else None
    )
    monkeypatch.setattr(adapters, "MAX_ACTION_RECOGNITION_CANDIDATES", 1)
    request = _request()
    request.include_candidates = True

    with _clip(frame_count=3, source_fps=10), pytest.raises(PayloadTooLargeError):
        _adapter(model).infer_from_request(request)

    assert len(model.calls) == (0 if known_estimate else 1)


@contextlib.contextmanager
def _clip(frame_count: int, source_fps: float):
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    with patch(f"{MODULE}.video_source_path") as source_path, patch(
        f"{MODULE}.probe_video", return_value=(source_fps, frame_count)
    ), patch(
        f"{MODULE}.read_frame_windows",
        side_effect=lambda path, windows, max_frame_side=None, frame_transform=None, **kwargs: (
            [frame_transform(frame) if frame_transform else frame] * len(window)
            for window in windows
        ),
    ):
        source_path.return_value.__enter__ = MagicMock(return_value="/tmp/clip")
        source_path.return_value.__exit__ = MagicMock(return_value=False)
        yield


@pytest.mark.parametrize("prepare_frames", [False, True])
def test_window_segments_map_to_clip_frame_indices(prepare_frames) -> None:
    model = _FakeModel(
        responses=[[ActionRecognitionPrediction(0, 15, "walk")]],
        class_names=["walk", "run"],
        sampling=VideoSampling(window_seconds=8.0, sample_fps=2.0, min_frames=4),
    )
    prepared = MagicMock(side_effect=lambda frame: frame)
    if prepare_frames:
        model.frame_storage_transform = prepared
    with _clip(frame_count=100, source_fps=10.0):
        response = _adapter(model).infer_from_request(_request())
    assert prepared.call_count == (2 if prepare_frames else 0)
    # 10 s against an 8 s window: one whole window plus the 2 s tail.
    assert response.windows_classified == 2
    assert response.source_fps == 10.0
    assert response.frame_count == 100
    # The model reported its own frames 0..15; the clip counts 0..75.
    assert response.timeline[0].start_frame_idx == 0
    assert response.timeline[0].end_frame_idx == 75
    assert response.timeline[0].class_name == "walk"
    assert response.timeline[0].class_id == 0


def test_clip_response_reports_the_loaded_package(monkeypatch) -> None:
    from inference.core.models import base

    monkeypatch.setattr(base, "USE_INFERENCE_MODELS", True)
    model = _FakeModel(responses=[[]], class_names=["walk"])
    model.resolved_model = SimpleNamespace(
        model_id="workspace/model",
        model_package_id="video-package",
        backend="torch",
        quantization="fp16",
    )
    with _clip(frame_count=3, source_fps=10.0):
        response = _adapter(model).infer_from_request(_request())

    assert response.model_dump()["resolved_model"] == vars(model.resolved_model)


def test_clip_response_reports_the_package_of_a_wrapped_reasoner(monkeypatch) -> None:
    from inference.core.models import base
    from inference.core.models import inference_models_adapters as adapters
    from inference_models.entities import ResolvedModelMetadata
    from inference_models.models.cosmos3.cosmos3_action_recognition import (
        Cosmos3EdgeActionRecognition,
    )
    from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner

    monkeypatch.setattr(base, "USE_INFERENCE_MODELS", True)
    # The registry hands a fine-tune over as a bare reasoner, and the
    # auto-loader stamps the package on that reasoner, not on the wrapper.
    reasoner = Cosmos3EdgeReasoner.__new__(Cosmos3EdgeReasoner)
    reasoner._processor = SimpleNamespace(tokenizer=None)
    reasoner.package_dir = None
    reasoner.prompt_video = MagicMock(return_value="")
    reasoner.resolved_model = ResolvedModelMetadata(
        model_id="workspace/model",
        model_package_id="video-package",
        backend="hugging-face",
        quantization="bf16",
    )

    with patch.object(adapters, "AutoModel") as auto_model:
        auto_model.from_pretrained.return_value = reasoner
        adapter = adapters.InferenceModelsActionRecognitionAdapter(
            model_id="workspace/model", api_key="key"
        )
    with _clip(frame_count=3, source_fps=10.0):
        response = adapter.infer_from_request(_request())

    assert isinstance(adapter._model, Cosmos3EdgeActionRecognition)
    assert adapter._model._reasoner is reasoner
    assert reasoner.prompt_video.call_count == 1
    assert response.model_dump()["resolved_model"] == {
        "model_id": "workspace/model",
        "model_package_id": "video-package",
        "backend": "hugging-face",
        "quantization": "bf16",
    }


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

    # 17 s against an 8 s window: two whole windows plus the 1 s tail.
    assert response.windows_classified == 3
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
        side_effect=lambda path, windows, max_frame_side, **kwargs: (
            [frame] for _ in windows
        ),
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


@pytest.mark.parametrize("reported_count", [31, 120])
@pytest.mark.parametrize("include_candidates", [False, True])
def test_early_decode_end_clips_timeline_candidates_and_metadata(
    reported_count, include_candidates
):
    class ScoredModel(_FakeModel):
        supports_confidence = True
        supports_observed_duration = True
        confidence_threshold = 0.5

        def infer(self, frames, class_names=None, fps=None, **kwargs):
            self.calls.append(kwargs)
            return [ActionRecognitionPrediction(0, 16, "walk", 0.9, True)]

    model = ScoredModel(
        [],
        ["walk"],
        VideoSampling(
            window_seconds=4,
            sample_fps=4,
            min_frames=1,
            max_frames=16,
            fixed_sample_fps=True,
        ),
    )
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.side_effect = [(True, frame)] * 30 + [(False, None)]
    request = _request()
    request.include_candidates = include_candidates

    with patch(f"{MODULE}.video_source_path") as source_path, patch(
        f"{MODULE}.probe_video", return_value=(30.0, reported_count)
    ), patch("inference.core.utils.video_utils.cv2.VideoCapture", return_value=capture):
        source_path.return_value.__enter__.return_value = "/tmp/clip"
        response = _adapter(model).infer_from_request(request)

    assert model.calls[0]["duration_seconds"] == 1.0
    assert response.frame_count == 30
    assert response.timeline[0].end_frame_idx == 29
    if include_candidates:
        assert response.candidates[0].end_frame_idx == 29
    capture.release.assert_called_once()


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


def test_a_clip_over_the_duration_cap_is_refused_before_any_window_runs(monkeypatch):
    import inference.core.models.inference_models_adapters as adapters
    from inference.core.exceptions import PayloadTooLargeError

    monkeypatch.setattr(adapters, "MAX_VIDEO_DURATION_SECONDS", 60.0)
    model = _FakeModel(responses=[[]], class_names=["walk"])

    # 61 s at 10 fps, one second past the cap.
    with _clip(frame_count=610, source_fps=10.0):
        with pytest.raises(PayloadTooLargeError, match="at most 60 s"):
            _adapter(model).infer_from_request(_request())

    assert model.calls == []


def test_a_clip_exactly_at_the_duration_cap_is_served(monkeypatch):
    import inference.core.models.inference_models_adapters as adapters

    monkeypatch.setattr(adapters, "MAX_VIDEO_DURATION_SECONDS", 60.0)
    model = _FakeModel(responses=[[]], class_names=["walk"])

    with _clip(frame_count=600, source_fps=10.0):
        response = _adapter(model).infer_from_request(_request())

    assert response.frame_count == 600


def test_a_negative_duration_cap_removes_the_limit(monkeypatch):
    import inference.core.models.inference_models_adapters as adapters

    monkeypatch.setattr(adapters, "MAX_VIDEO_DURATION_SECONDS", -1)
    model = _FakeModel(responses=[[]], class_names=["walk"])

    with _clip(frame_count=36000, source_fps=10.0):
        response = _adapter(model).infer_from_request(_request())

    assert response.frame_count == 36000


def _side_handed_to_the_reader(model) -> object:
    with patch(f"{MODULE}.video_source_path") as source_path, patch(
        f"{MODULE}.probe_video", return_value=(10.0, 100)
    ), patch(
        f"{MODULE}.read_frame_windows", return_value=(item for item in [])
    ) as reader:
        source_path.return_value.__enter__ = MagicMock(return_value="/tmp/clip")
        source_path.return_value.__exit__ = MagicMock(return_value=False)
        _adapter(model).infer_from_request(_request())
    return reader.call_args.kwargs["max_frame_side"]


def test_cancellation_after_one_window_closes_decoder_and_source(monkeypatch):
    from inference.core.models import inference_models_adapters as adapters
    from inference.core.utils.video_processing import (
        VideoProcessingCancelledError,
        VideoProcessingControl,
    )

    state = {"cancelled": False, "source_closed": False}

    @contextlib.contextmanager
    def source(**kwargs):
        try:
            yield "/tmp/clip"
        finally:
            state["source_closed"] = True

    class CancellingModel(_FakeModel):
        def infer(self, frames, **kwargs):
            self.calls.append(kwargs)
            state["cancelled"] = True
            return []

    model = CancellingModel([], ["walk"], VideoSampling(window_seconds=4, sample_fps=4))
    capture = MagicMock()
    capture.isOpened.return_value = True
    capture.read.return_value = (True, np.zeros((8, 8, 3), dtype=np.uint8))
    control = VideoProcessingControl(
        timeout_seconds=60, is_disconnected=lambda: state["cancelled"]
    )
    monkeypatch.setattr(adapters, "video_source_path", source)
    monkeypatch.setattr(adapters, "probe_video", lambda **kwargs: (4.0, 128))

    with patch(
        "inference.core.utils.video_utils.cv2.VideoCapture", return_value=capture
    ):
        with pytest.raises(VideoProcessingCancelledError):
            _adapter(model).infer_from_request(_request(), processing_control=control)

    assert len(model.calls) == 1
    assert state["source_closed"]
    capture.release.assert_called_once()


def test_an_untrained_model_is_read_at_the_1080p_ceiling() -> None:
    model = _FakeModel(responses=[], sampling=VideoSampling())

    assert _side_handed_to_the_reader(model) == 1920


def test_a_trained_model_without_a_declared_side_is_read_whole() -> None:
    model = _FakeModel(responses=[], sampling=VideoSampling(max_frames=64))

    assert _side_handed_to_the_reader(model) is None


def test_load_action_recognition_model_passes_the_zero_shot_id_through(
    monkeypatch,
) -> None:
    from inference.core.models import inference_models_adapters as adapters

    monkeypatch.setattr(adapters, "VJEPA2_1_ENABLED", True)

    with patch.object(adapters, "AutoModel") as auto_model, patch.object(
        adapters,
        "_as_action_recognition_model",
        side_effect=lambda model, model_id: model,
    ):
        adapters.load_action_recognition_model(
            model_id="nvidia/cosmos-3-edge-action-recognition", api_key="key"
        )

    assert (
        auto_model.from_pretrained.call_args.kwargs["model_id_or_path"]
        == "nvidia/cosmos-3-edge-action-recognition"
    )


@pytest.mark.parametrize(
    "enabled, model_type, blocked",
    [
        (True, "vjepa2-1-vitb-384", False),
        (True, "vjepa2_1", False),
        (False, "vjepa2-1-vitb-384", True),
        (False, "vjepa2_1", True),
        (False, "cosmos-3-edge", False),
    ],
)
def test_shared_loader_honors_vjepa_enablement(
    monkeypatch, enabled, model_type, blocked
) -> None:
    from inference.core.exceptions import ModelDeploymentNotSupportedError
    from inference.core.models import inference_models_adapters as adapters
    from inference.core.registries import roboflow

    monkeypatch.setattr(adapters, "VJEPA2_1_ENABLED", enabled)
    with patch.object(
        roboflow, "get_model_type", return_value=("action-recognition", model_type)
    ) as metadata, patch.object(adapters, "AutoModel") as auto_model, patch.object(
        adapters,
        "_as_action_recognition_model",
        side_effect=lambda model, model_id: model,
    ):
        if blocked:
            with pytest.raises(ModelDeploymentNotSupportedError, match="disabled"):
                adapters.load_action_recognition_model("project/2", api_key="key")
            auto_model.from_pretrained.assert_not_called()
        else:
            adapters.load_action_recognition_model("project/2", api_key="key")
            auto_model.from_pretrained.assert_called_once()
        if enabled:
            metadata.assert_not_called()
        else:
            metadata.assert_called_once_with(
                model_id="project/2",
                api_key="key",
                countinference=None,
                service_secret=None,
            )
