"""Exercise local-video orchestration with real decoding and CPU model contracts."""

import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
import torch

from inference_models import (
    ActionRecognitionPrediction,
    ActionRecognitionVideoPipeline,
    VideoInputError,
    VideoProcessingCancelledError,
    VideoProcessingControl,
    VideoProcessingLimitError,
    VideoProcessingLimits,
    VideoProcessingTimeoutError,
    VideoSampling,
)
from inference_models.models.base.action_recognition import plan_windows
from inference_models.video import io as video_io


class WindowModel:
    supports_confidence = True
    supports_observed_duration = True
    confidence_threshold = 0.9
    class_names = ["walk", "run"]
    span_semantics = "class_union"
    recommended_parameters = SimpleNamespace(
        confidence=0.9,
        per_class_confidence={"walk": 0.9, "run": 0.5},
    )

    def __init__(self, *, scored=True, sampling=None):
        self.supports_confidence = scored
        self.supports_observed_duration = scored
        self.span_semantics = "class_union" if scored else "instances"
        self.video_sampling = sampling or VideoSampling(
            window_seconds=1.0,
            sample_fps=4.0,
            min_frames=1,
            max_frames=4,
            overlap_frames=2 if scored else 0,
            end_aligned=scored,
            fixed_sample_fps=scored,
        )
        self.calls = []
        self.prepared = []
        self.frame_storage_transform = self.prepare

    def prepare(self, frame):
        self.prepared.append(int(frame[0, 0, 2]))
        return frame[:8, :8].copy()

    def estimate_candidate_count(self, sampled_frames, *, class_names=None):
        return 2 * sampled_frames if self.supports_confidence else None

    def infer(self, frames, class_names=None, fps=None, **kwargs):
        self.calls.append((frames, fps, kwargs))
        if not self.supports_confidence:
            return [ActionRecognitionPrediction(0, len(frames) - 1, "walk")]
        option = kwargs["confidence"]
        predictions = []
        for label, score in zip(self.class_names, torch.tensor([0.9, 0.6])):
            threshold = (
                self.recommended_parameters.per_class_confidence[label]
                if option == "best"
                else (self.confidence_threshold if option == "default" else option)
            )
            if score >= threshold and (class_names is None or label in class_names):
                predictions.append(
                    ActionRecognitionPrediction(
                        0, len(frames), label, score.item(), True
                    )
                )
        return predictions


def write_clip(path, *, count=30, fps=10.0):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (16, 16))
    assert writer.isOpened()
    try:
        for number in range(count):
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[:, :, 0] = number % 256
            writer.write(frame)
    finally:
        writer.release()
    return path


@pytest.fixture
def clip(tmp_path):
    return write_clip(tmp_path / "clip.avi")


@pytest.fixture
def captures(monkeypatch):
    real_capture = cv2.VideoCapture
    opened = []

    def open_capture(*args, **kwargs):
        capture = MagicMock(wraps=real_capture(*args, **kwargs))
        opened.append(capture)
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", open_capture)
    return opened


@pytest.mark.parametrize("scored", [False, True])
@pytest.mark.parametrize("confidence", [None, 0.9, "best"])
@pytest.mark.parametrize("include_candidates", [False, True])
def test_full_video_preserves_sampling_scoring_and_coordinates(
    clip, scored, confidence, include_candidates
):
    model = WindowModel(scored=scored)
    result = ActionRecognitionVideoPipeline(model).run(
        clip,
        confidence=confidence,
        include_candidates=include_candidates,
    )
    plans = plan_windows(30, 10.0, model.video_sampling)
    assert result.frame_count == 30
    assert result.source_fps == 10.0
    assert result.windows_classified == len(plans)
    assert len(model.calls) == len(plans)
    assert model.prepared == sorted({i for p in plans for i in p.frame_indices})
    for (frames, fps, kwargs), plan in zip(model.calls, plans):
        assert [int(frame[0, 0, 2]) for frame in frames] == list(plan.frame_indices)
        assert all(frame.shape == (8, 8, 3) for frame in frames)
        assert fps == plan.sample_fps
        if not scored:
            assert "confidence" not in kwargs
            assert "duration_seconds" not in kwargs
    assert result.timeline[0].start_frame_idx == 0
    assert result.timeline[0].end_frame_idx == (29 if scored else 28)
    assert [p.class_name for p in result.timeline] == (
        ["walk", "run"] if scored and confidence == "best" else ["walk"]
    )
    assert (result.candidates is not None) == (scored and include_candidates)
    if scored and include_candidates:
        assert len(result.candidates) == 2 * len(plans)
    if not scored:
        assert result.confidence_threshold is None
        assert result.per_class_confidence_thresholds is None
        assert result.timeline[0].confidence is None


@pytest.mark.parametrize("confidence", ["best", "default", 0.9, 1, 0])
def test_candidates_do_not_change_filtered_timeline(clip, confidence):
    results = [
        ActionRecognitionVideoPipeline(WindowModel()).run(
            clip,
            confidence=confidence,
            include_candidates=include,
        )
        for include in (False, True)
    ]
    assert results[0].timeline == results[1].timeline


def test_class_filter_restricts_predictions_without_renumbering(clip):
    result = ActionRecognitionVideoPipeline(WindowModel()).run(
        clip,
        class_names=["run"],
        confidence="best",
        include_candidates=True,
    )
    assert [(p.class_name, p.class_id) for p in result.timeline] == [("run", 1)]
    assert all(p.class_name == "run" for p in result.candidates)


@pytest.mark.parametrize("count,fps", [(2, 30.0), (3, 10.0), (31, 29.97), (47, 29.97)])
def test_short_and_fractional_tail_windows_stay_within_source(tmp_path, count, fps):
    path = write_clip(tmp_path / "short.avi", count=count, fps=fps)
    model = WindowModel()
    result = ActionRecognitionVideoPipeline(model).run(path, include_candidates=True)
    assert result.windows_classified >= 1
    assert all(
        0 <= p.start_frame_idx <= p.end_frame_idx < count
        for p in result.timeline + result.candidates
    )
    assert result.timeline[0].end_frame_idx == count - 1
    assert all(call[2]["duration_seconds"] <= 1.0 for call in model.calls)


def test_observed_eof_corrects_metadata_and_spans(clip, monkeypatch, captures):
    monkeypatch.setattr(
        "inference_models.video.action_recognition.probe_video",
        lambda **kwargs: (10.0, 43),
    )
    model = WindowModel()
    result = ActionRecognitionVideoPipeline(model).run(clip, include_candidates=True)
    assert result.frame_count == 30
    assert result.windows_processed > result.windows_classified
    assert all(p.end_frame_idx < 30 for p in result.timeline + result.candidates)
    assert all(capture.release.call_count == 1 for capture in captures)


def test_stream_events_merge_to_same_result_without_retaining_history(clip):
    pipeline = ActionRecognitionVideoPipeline(WindowModel())
    progress = []
    timeline, candidates = [], []
    with pipeline.stream(
        clip, include_candidates=True, on_progress=progress.append
    ) as events:
        for event in events:
            assert event.progress == progress[-1]
            event.merge_into(timeline)
            candidates.extend(event.candidates)
    collected = ActionRecognitionVideoPipeline(WindowModel()).run(
        clip, include_candidates=True
    )
    assert (
        sorted(timeline, key=lambda p: (p.start_frame_idx, p.class_id))
        == collected.timeline
    )
    assert candidates == collected.candidates
    assert progress[-1].windows_processed == progress[-1].windows_total


@pytest.mark.parametrize("failure", ["break", "sink", "model", "progress", "cancel"])
def test_early_exit_releases_decoder(clip, captures, failure):
    model = WindowModel()
    control = (
        VideoProcessingControl(is_cancelled=lambda: len(model.calls) >= 1)
        if failure == "cancel"
        else None
    )
    if failure == "model":
        model.infer = MagicMock(side_effect=RuntimeError("model failed"))

    def progress(_):
        if failure == "progress":
            raise RuntimeError("progress failed")

    pipeline = ActionRecognitionVideoPipeline(model)
    if failure == "break":
        with pipeline.stream(clip) as events:
            next(events)
    else:
        error = VideoProcessingCancelledError if failure == "cancel" else RuntimeError
        with pytest.raises(error):
            with pipeline.stream(clip, control=control, on_progress=progress) as events:
                for _ in events:
                    if failure == "sink":
                        raise RuntimeError("sink failed")
    assert captures
    assert all(capture.release.call_count == 1 for capture in captures)


def test_deadline_stops_before_opening_video(clip, captures, monkeypatch):
    monkeypatch.setattr("inference_models.video.control.monotonic", lambda: 0.0)
    control = VideoProcessingControl(timeout_seconds=1.0)
    monkeypatch.setattr("inference_models.video.control.monotonic", lambda: 1.0)
    with pytest.raises(VideoProcessingTimeoutError):
        ActionRecognitionVideoPipeline(WindowModel()).run(clip, control=control)
    assert captures == []


@pytest.mark.parametrize(
    "changes",
    [
        {"max_windows": 1},
        {"max_sampled_frames": 3},
        {"max_candidates": 1},
        {"max_retained_frame_bytes": 1},
        {"max_result_bytes": 1},
    ],
)
def test_resource_limits_raise_and_release_decoder(clip, changes, captures):
    limits = replace(VideoProcessingLimits(), **changes)
    pipeline = ActionRecognitionVideoPipeline(WindowModel(), limits=limits)
    with pytest.raises(VideoProcessingLimitError):
        pipeline.run(clip, include_candidates=True)
    assert all(capture.release.call_count == 1 for capture in captures)


def test_timeline_limit_applies_after_merging(clip):
    limits = VideoProcessingLimits(max_timeline_entries=1)
    pipeline = ActionRecognitionVideoPipeline(WindowModel(), limits=limits)
    assert len(pipeline.run(clip).timeline) == 1
    with pytest.raises(VideoProcessingLimitError, match="timeline"):
        pipeline.run(clip, confidence="best")


def test_streaming_can_exceed_total_candidate_collection_budget(clip):
    limits = VideoProcessingLimits(max_candidates=8)
    pipeline = ActionRecognitionVideoPipeline(WindowModel(), limits=limits)
    with pipeline.stream(clip, include_candidates=True) as events:
        assert sum(len(event.candidates) for event in events) == 10
    with pytest.raises(VideoProcessingLimitError, match="candidates"):
        pipeline.run(clip, include_candidates=True)


def test_huge_window_is_rejected_before_allocating_indices():
    sampling = VideoSampling(window_seconds=1e9, sample_fps=1e9, min_frames=1)
    with pytest.raises(VideoProcessingLimitError, match="sampled-frame"):
        plan_windows(10**12, 30.0, sampling, max_sampled_frames=10)


def test_huge_overlapping_plan_is_rejected_before_allocating_starts():
    sampling = VideoSampling(window_seconds=1, sample_fps=1e6, overlap_frames=999999)
    with pytest.raises(VideoProcessingLimitError, match="windows"):
        plan_windows(10**12, 30.0, sampling, max_windows=10)


def test_probe_cancellation_during_fallback_count_releases_both_captures(
    clip, monkeypatch, captures
):
    original = cv2.VideoCapture

    def capture_without_count(*args, **kwargs):
        capture = original(*args, **kwargs)
        original_get = capture.get
        capture.get = lambda prop: (
            0 if prop == cv2.CAP_PROP_FRAME_COUNT else original_get(prop)
        )
        return capture

    monkeypatch.setattr(cv2, "VideoCapture", capture_without_count)

    def check():
        if len(captures) == 2:
            raise VideoProcessingCancelledError("cancel")

    with pytest.raises(VideoProcessingCancelledError):
        video_io.probe_video(clip, check_cancelled=check)
    assert len(captures) == 2
    assert all(capture.release.call_count == 1 for capture in captures)


def test_local_path_rejects_url_without_opening_decoder(captures):
    with pytest.raises(VideoInputError):
        ActionRecognitionVideoPipeline(WindowModel()).run(
            "https://example.com/video.mp4"
        )
    assert captures == []


def test_public_api_wraps_an_automodel_cosmos_reasoner(clip):
    from inference_models.models.cosmos3.cosmos3_action_recognition import (
        Cosmos3EdgeActionRecognition,
    )
    from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner

    reasoner = Cosmos3EdgeReasoner.__new__(Cosmos3EdgeReasoner)
    reasoner._processor = SimpleNamespace(tokenizer=None)
    reasoner.package_dir = None
    reasoner.prompt_video = MagicMock(return_value="")
    pipeline = ActionRecognitionVideoPipeline(reasoner)
    assert isinstance(pipeline.model, Cosmos3EdgeActionRecognition)
    result = pipeline.run(
        clip, confidence="ignored for Cosmos", include_candidates=True
    )
    assert result.windows_classified == 1
    assert result.candidates is None
    reasoner.prompt_video.assert_called_once()


@pytest.mark.parametrize("bad", [True, -0.1, 1.1, float("nan"), "unknown"])
def test_scored_confidence_is_validated(clip, bad):
    with pytest.raises(ValueError, match="confidence"):
        ActionRecognitionVideoPipeline(WindowModel()).run(clip, confidence=bad)


def test_package_only_subprocess_blocks_server_workflows_and_streamvision(
    clip, tmp_path
):
    package_root = Path(__file__).resolve().parents[3]
    script = """
import importlib.abc
import sys
class NoServerImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'inference', 'roboflow_workflows', 'streamvision'}:
            raise AssertionError('Forbidden import: ' + fullname)
sys.meta_path.insert(0, NoServerImports())
from inference_models import ActionRecognitionVideoPipeline, ActionRecognitionPrediction, VideoSampling
class Model:
    video_sampling = VideoSampling(window_seconds=1, sample_fps=4)
    supports_confidence = False
    supports_observed_duration = False
    class_names = ['walk']
    def infer(self, frames, **kwargs):
        return [ActionRecognitionPrediction(0, len(frames) - 1, 'walk')]
result = ActionRecognitionVideoPipeline(Model()).run(sys.argv[1])
assert result.frame_count == 30
assert result.timeline[0].class_name == 'walk'
print('isolated local video passed')
"""
    environment = dict(os.environ, PYTHONPATH=str(package_root))
    result = subprocess.run(
        [sys.executable, "-c", script, str(clip)],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "isolated local video passed" in result.stdout
