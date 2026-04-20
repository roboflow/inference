"""Unit tests for the SegmentAnything2Video model wrapper.

These tests mock out the ``sam2`` package so we can exercise the
per-video session management logic (prompt/track/reset) without
needing SAM2 weights or the ``sam2`` library installed.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


def _install_sam2_mocks():
    """Install fake ``sam2`` submodules into ``sys.modules``.

    Covers every submodule imported by ``inference.models.sam2`` — the
    image model path (``SAM2ImagePredictor``, ``build_sam2``) is mocked
    even though we only exercise the video path, because importing
    ``inference.models.sam2`` pulls in the image module too.
    """
    sam2_mock = MagicMock()
    sam2_mock.utils = MagicMock()
    sam2_mock.utils.misc = MagicMock()
    sam2_mock.utils.misc.get_sdp_backends = lambda z: []
    sam2_mock.build_sam = MagicMock()
    sam2_mock.sam2_camera_predictor = MagicMock()
    sam2_mock.sam2_image_predictor = MagicMock()

    build_mock = MagicMock()
    # Each call returns a new MagicMock representing a distinct predictor.
    build_mock.side_effect = lambda *args, **kwargs: MagicMock(name="SAM2CameraPredictor")
    sam2_mock.build_sam.build_sam2_camera_predictor = build_mock
    sam2_mock.build_sam.build_sam2 = MagicMock(
        side_effect=lambda *a, **k: MagicMock(name="SAM2Model")
    )

    predictor_cls_mock = MagicMock()
    sam2_mock.sam2_camera_predictor.SAM2CameraPredictor = predictor_cls_mock
    sam2_mock.sam2_image_predictor.SAM2ImagePredictor = MagicMock()

    for name, mod in [
        ("sam2", sam2_mock),
        ("sam2.utils", sam2_mock.utils),
        ("sam2.utils.misc", sam2_mock.utils.misc),
        ("sam2.build_sam", sam2_mock.build_sam),
        ("sam2.sam2_camera_predictor", sam2_mock.sam2_camera_predictor),
        ("sam2.sam2_image_predictor", sam2_mock.sam2_image_predictor),
    ]:
        sys.modules[name] = mod

    return predictor_cls_mock, build_mock


_predictor_cls_mock, _build_mock = _install_sam2_mocks()

# Short-circuit the RoboflowCoreModel weight download path so the
# constructor doesn't try to reach the Roboflow API.  Imported eagerly
# before the patches so ``inference.core.models.roboflow`` exists and
# can be targeted.
from inference.core.models import roboflow as _roboflow_module  # noqa: E402

with patch.object(
    _roboflow_module.RoboflowCoreModel, "download_weights", return_value=None
), patch.object(
    _roboflow_module.RoboflowInferenceModel,
    "cache_file",
    return_value="/tmp/fake_sam2_weights.pt",
):
    from inference.models.sam2.segment_anything2_video import (  # noqa: E402
        SegmentAnything2Video,
    )


def _make_fake_predictor():
    """Build a fake ``SAM2CameraPredictor`` that returns unpackable
    tuples from the two methods the wrapper actually calls."""
    import torch

    predictor = MagicMock(name="SAM2CameraPredictor")

    def _add_new_prompt(frame_idx, obj_id, bbox, clear_old_points, normalize_coords):
        mask_logits = torch.ones(1, 1, 120, 160)
        return frame_idx, [obj_id], mask_logits

    predictor.add_new_prompt.side_effect = _add_new_prompt
    predictor.track.side_effect = lambda frame: (
        [0],
        torch.ones(1, 1, frame.shape[0], frame.shape[1]),
    )
    return predictor


@pytest.fixture
def fake_model():
    _build_mock.reset_mock()
    _build_mock.side_effect = lambda *args, **kwargs: _make_fake_predictor()
    with patch.object(
        _roboflow_module.RoboflowCoreModel, "download_weights", return_value=None
    ), patch.object(
        _roboflow_module.RoboflowInferenceModel,
        "cache_file",
        return_value="/tmp/fake_sam2_weights.pt",
    ):
        yield SegmentAnything2Video(model_id="sam2/hiera_tiny")


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_unknown_version_id_rejected():
    with patch.object(
        _roboflow_module.RoboflowCoreModel, "download_weights", return_value=None
    ), patch.object(
        _roboflow_module.RoboflowInferenceModel,
        "cache_file",
        return_value="/tmp/fake_sam2_weights.pt",
    ):
        with pytest.raises(ValueError, match="Unknown SAM2 version_id"):
            SegmentAnything2Video(model_id="sam2/not_a_version")


def test_has_session_false_for_unknown_video_id(fake_model):
    assert fake_model.has_session("video-A") is False


def test_prompt_and_track_creates_session_and_returns_masks(fake_model):
    # Wire the fake camera predictor's add_new_prompt to return a
    # plausible response tuple.
    def _fake_add_new_prompt(**kwargs):
        import torch

        # Return (frame_idx, object_ids, mask_logits) as SAM2CameraPredictor does.
        mask_logits = torch.ones(1, 1, 120, 160)
        return 0, [kwargs["obj_id"]], mask_logits

    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    boxes = [(10.0, 10.0, 60.0, 60.0), (80.0, 80.0, 120.0, 120.0)]

    masks, obj_ids = fake_model.prompt_and_track(
        video_id="video-A",
        frame=frame,
        boxes_xyxy=boxes,
        clear_old_prompts=True,
    )

    assert fake_model.has_session("video-A") is True
    assert masks.dtype == bool
    assert masks.shape[1:] == (120, 160)
    # SAM2 camera predictor is mocked so we don't assert exact obj ids,
    # only that the returned arrays align.
    assert len(obj_ids) == masks.shape[0]


def test_prompt_and_track_with_no_boxes_returns_empty(fake_model):
    frame = np.zeros((50, 80, 3), dtype=np.uint8)
    masks, obj_ids = fake_model.prompt_and_track(
        video_id="video-A",
        frame=frame,
        boxes_xyxy=[],
        clear_old_prompts=True,
    )
    assert masks.shape == (0, 50, 80)
    assert obj_ids.shape == (0,)


def test_track_without_prior_prompt_raises(fake_model):
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="No SAM2 video session"):
        fake_model.track(video_id="unknown", frame=frame)


def test_reset_session_removes_state(fake_model):
    def _fake_add_new_prompt(**kwargs):
        import torch

        return 0, [0], torch.ones(1, 1, 50, 80)

    frame = np.zeros((50, 80, 3), dtype=np.uint8)
    fake_model.prompt_and_track(
        video_id="video-A",
        frame=frame,
        boxes_xyxy=[(1.0, 1.0, 10.0, 10.0)],
        clear_old_prompts=True,
    )
    assert fake_model.has_session("video-A") is True

    fake_model.reset_session("video-A")
    assert fake_model.has_session("video-A") is False


def test_separate_video_ids_get_separate_predictors(fake_model):
    frame = np.zeros((50, 80, 3), dtype=np.uint8)

    # First prompt for video-A and video-B — each should trigger a
    # ``build_sam2_camera_predictor`` call (one predictor per video).
    fake_model.prompt_and_track(
        video_id="video-A",
        frame=frame,
        boxes_xyxy=[(1.0, 1.0, 10.0, 10.0)],
        clear_old_prompts=True,
    )
    fake_model.prompt_and_track(
        video_id="video-B",
        frame=frame,
        boxes_xyxy=[(1.0, 1.0, 10.0, 10.0)],
        clear_old_prompts=True,
    )

    assert _build_mock.call_count == 2
