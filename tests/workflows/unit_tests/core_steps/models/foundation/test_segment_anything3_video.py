"""Unit tests for the SAM3 Video Tracker workflow block."""

import sys
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3_video import (
    v1 as sam3_video_module,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_video.v1 import (
    BlockManifest,
    SegmentAnything3VideoBlockV1,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_video.v1_tensor import (
    BlockManifest as TensorBlockManifest,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_video.v1_tensor import (
    SegmentAnything3VideoBlockV1 as TensorSegmentAnything3VideoBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_frame(
    video_id: str = "stream-0",
    frame_number: int = 0,
    shape=(120, 160, 3),
) -> WorkflowImageData:
    image = np.zeros(shape, dtype=np.uint8)
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        fps=30,
        frame_timestamp=datetime(2024, 1, 1, 0, 0, frame_number % 60),
    )
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"{video_id}:{frame_number}"),
        numpy_image=image,
        video_metadata=metadata,
    )


@dataclass(frozen=True)
class _FakeFrameResult:
    """Mirrors ``inference_models``' ``SAM3VideoFrameResult`` fields."""

    masks: np.ndarray
    object_ids: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    prompt_to_object_ids: Dict[str, List[int]]
    state_dict: dict


class _FakeConceptModel:
    """Stand-in for an ``inference_models.SAM3Video`` instance.

    Each registered concept yields one object with a descending score
    (0.9, 0.8, ...).  From the third processed frame of a session, a
    second instance of the first concept appears, emulating mid-stream
    detection without re-prompting.
    """

    def __init__(self):
        self.calls = []
        self._session_counter = 0

    def prompt(self, image, text, state_dict=None, clear_old_prompts=True):
        self._session_counter += 1
        session = {"id": self._session_counter, "prompts": list(text), "frames": 0}
        self.calls.append(("prompt", {"text": list(text)}))
        return self._step(image, session)

    def track(self, image, state_dict=None):
        assert state_dict is not None, "block must thread state into track"
        session = state_dict["session"]
        self.calls.append(("track", {"session_id": session["id"]}))
        return self._step(image, session)

    def _step(self, image, session) -> _FakeFrameResult:
        session["frames"] += 1
        prompts = session["prompts"]
        object_ids = list(range(len(prompts)))
        prompt_to_object_ids = {p: [i] for i, p in enumerate(prompts)}
        if session["frames"] >= 3 and prompts:
            new_id = len(object_ids)
            object_ids.append(new_id)
            prompt_to_object_ids[prompts[0]].append(new_id)
        n = len(object_ids)
        h, w = image.shape[:2]
        masks = np.zeros((n, h, w), dtype=bool)
        for i in range(n):
            masks[i, i, i] = True
        return _FakeFrameResult(
            masks=masks,
            object_ids=np.asarray(object_ids, dtype=np.int64),
            scores=np.asarray([0.9 - 0.1 * i for i in range(n)], dtype=np.float32),
            boxes=np.tile(
                np.asarray([[0.0, 0.0, 10.0, 10.0]], dtype=np.float32), (n, 1)
            ),
            prompt_to_object_ids=prompt_to_object_ids,
            state_dict={"session": session},
        )


class _FakeVisualModel:
    """Stand-in for the point- and box-prompted SAM3 tracker."""

    def __init__(self):
        self.calls = []

    @staticmethod
    def _result(image, object_count):
        height, width = image.shape[:2]
        masks = np.zeros((object_count, height, width), dtype=bool)
        for index in range(object_count):
            masks[index, index, index] = True
        return (
            masks,
            np.arange(object_count, dtype=np.int64),
            {"object_count": object_count},
        )

    def prompt(
        self,
        image,
        bboxes,
        points,
        state_dict=None,
        clear_old_prompts=True,
        frame_idx=0,
    ):
        self.calls.append(
            (
                "prompt",
                {
                    "bboxes": list(bboxes),
                    "points": list(points),
                    "frame_idx": frame_idx,
                    "had_prior_state": state_dict is not None,
                },
            )
        )
        object_count = len(bboxes) + (1 if points else 0)
        return self._result(image=image, object_count=object_count)

    def track(self, image, state_dict=None):
        assert state_dict is not None, "block must thread state into track"
        self.calls.append(("track", {"had_prior_state": True}))
        return self._result(
            image=image,
            object_count=state_dict["object_count"],
        )


def _make_block_with_fake_model():
    block = SegmentAnything3VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    fake = _FakeConceptModel()
    block._model = fake
    block._current_model_id = "sam3video"
    return block, fake


def _make_visual_block_with_fake_model():
    block = SegmentAnything3VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    fake = _FakeVisualModel()
    block._model = fake
    block._current_model_id = "sam3trackervideo"
    return block, fake


def _make_box_detections() -> sv.Detections:
    detections = sv.Detections(
        xyxy=np.asarray([[10, 20, 80, 90]], dtype=np.float32),
        confidence=np.asarray([0.91], dtype=np.float32),
        class_id=np.asarray([7], dtype=int),
    )
    detections.data["class_name"] = np.asarray(["vehicle"], dtype=object)
    detections.data["detection_id"] = np.asarray(["det-0"], dtype=object)
    return detections


def _run_single(block, frame, class_names=("person",), threshold=0.0):
    return block.run(
        images=[frame],
        class_names=list(class_names),
        model_id="sam3video",
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def test_manifest_parses_valid_config():
    data = {
        "type": "roboflow_core/sam3_video@v1",
        "name": "sam3_video_step",
        "images": "$inputs.image",
        "class_names": ["person", "forklift"],
    }
    manifest = BlockManifest.model_validate(data)
    assert manifest.type == "roboflow_core/sam3_video@v1"
    assert manifest.model_id == "sam3video"
    assert manifest.class_names == ["person", "forklift"]
    assert manifest.threshold == 0.5


def test_manifest_requires_class_names():
    with pytest.raises(Exception):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/sam3_video@v1",
                "name": "sam3_video_step",
                "images": "$inputs.image",
            }
        )


def test_manifest_selects_visual_model_for_point_prompts():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "sam3_video_step",
            "images": "$inputs.image",
            "tracking_mode": "visual",
            "points": [{"x": 10, "y": 20, "positive": True}],
        }
    )

    assert manifest.class_names is None
    assert manifest.model_id == "sam3trackervideo"


def test_manifest_updates_builtin_model_when_mode_changes():
    concept_manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "sam3_video_step",
            "images": "$inputs.image",
            "tracking_mode": "concept",
            "class_names": ["person"],
            "model_id": "sam3trackervideo",
        }
    )
    visual_manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "sam3_video_step",
            "images": "$inputs.image",
            "tracking_mode": "visual",
            "boxes": "$steps.detector.predictions",
            "model_id": "sam3video",
        }
    )

    assert concept_manifest.model_id == "sam3video"
    assert visual_manifest.model_id == "sam3trackervideo"


def test_manifest_requires_visual_prompts():
    with pytest.raises(Exception, match="Visual mode requires"):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/sam3_video@v1",
                "name": "sam3_video_step",
                "images": "$inputs.image",
                "tracking_mode": "visual",
            }
        )


def test_tensor_manifest_selects_visual_model_for_point_prompts():
    manifest = TensorBlockManifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "sam3_video_step",
            "images": "$inputs.image",
            "tracking_mode": "visual",
            "points": [{"x": 10, "y": 20, "positive": True}],
        }
    )

    assert manifest.model_id == "sam3trackervideo"


# ---------------------------------------------------------------------------
# Remote execution mode
# ---------------------------------------------------------------------------


def test_block_accepts_remote_execution_mode_at_initialisation():
    block = SegmentAnything3VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    assert block._step_execution_mode is StepExecutionMode.REMOTE


def test_block_rejects_remote_execution_mode_at_runtime():
    block = SegmentAnything3VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    with pytest.raises(NotImplementedError, match="LOCAL workflow step execution"):
        block.run(
            images=[],
            class_names=[],
            model_id="sam3video",
            threshold=0.5,
        )


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def test_model_loader_forwards_extra_weight_provider_headers(monkeypatch):
    from_pretrained = MagicMock(return_value=object())
    headers = {"x-temporary-auth-token": "token"}
    monkeypatch.setitem(
        sys.modules,
        "inference_models",
        SimpleNamespace(AutoModel=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    monkeypatch.setattr(
        sam3_video_module,
        "get_extra_weights_provider_headers",
        MagicMock(return_value=headers),
    )
    block = SegmentAnything3VideoBlockV1(
        model_manager=MagicMock(),
        api_key="rf-test",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    block._get_model(model_id="sam3video")

    from_pretrained.assert_called_once_with(
        model_id_or_path="sam3video",
        api_key="rf-test",
        weights_provider_extra_headers=headers,
    )


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------


def test_prompts_once_then_tracks():
    block, fake = _make_block_with_fake_model()

    for frame_number in range(3):
        _run_single(block, _make_frame(frame_number=frame_number))

    assert [c[0] for c in fake.calls] == ["prompt", "track", "track"]


def test_class_names_change_reseeds_session():
    block, fake = _make_block_with_fake_model()

    _run_single(block, _make_frame(frame_number=0), class_names=("person",))
    _run_single(block, _make_frame(frame_number=1), class_names=("person",))
    _run_single(block, _make_frame(frame_number=2), class_names=("person", "dog"))

    assert [c[0] for c in fake.calls] == ["prompt", "track", "prompt"]
    assert fake.calls[-1][1]["text"] == ["person", "dog"]


def test_frame_rollback_reseeds_session():
    block, fake = _make_block_with_fake_model()

    _run_single(block, _make_frame(frame_number=10))
    _run_single(block, _make_frame(frame_number=11))
    # Stream restarted from frame 0
    _run_single(block, _make_frame(frame_number=0))

    assert [c[0] for c in fake.calls] == ["prompt", "track", "prompt"]


def test_independent_sessions_per_video():
    block, fake = _make_block_with_fake_model()

    _run_single(block, _make_frame(video_id="cam-a", frame_number=0))
    _run_single(block, _make_frame(video_id="cam-b", frame_number=0))
    _run_single(block, _make_frame(video_id="cam-a", frame_number=1))
    _run_single(block, _make_frame(video_id="cam-b", frame_number=1))

    assert [c[0] for c in fake.calls] == ["prompt", "prompt", "track", "track"]
    # Each video keeps its own session
    track_sessions = {c[1]["session_id"] for c in fake.calls if c[0] == "track"}
    assert track_sessions == {1, 2}


def test_empty_class_names_yields_empty_predictions_without_model_calls():
    block, fake = _make_block_with_fake_model()

    result = block.run(
        images=[_make_frame()],
        class_names=[],
        model_id="sam3video",
        threshold=0.0,
    )

    assert fake.calls == []
    assert len(result[0]["predictions"]) == 0


# ---------------------------------------------------------------------------
# Output contents
# ---------------------------------------------------------------------------


def test_predictions_carry_class_labels_scores_and_tracker_ids():
    block, _ = _make_block_with_fake_model()

    result = _run_single(
        block, _make_frame(), class_names=("person", "dog"), threshold=0.0
    )
    predictions = result[0]["predictions"]

    assert len(predictions) == 2
    assert list(predictions.data["class_name"]) == ["person", "dog"]
    assert predictions.class_id.tolist() == [0, 1]
    assert predictions.tracker_id.tolist() == [0, 1]
    assert predictions.confidence.tolist() == pytest.approx([0.9, 0.8])
    assert predictions.mask.shape == (2, 120, 160)
    assert all(predictions.data["detection_id"])
    assert predictions.data["prediction_type"][0] == "instance-segmentation"


def test_objects_appearing_mid_stream_inherit_their_prompts_label():
    block, _ = _make_block_with_fake_model()

    _run_single(block, _make_frame(frame_number=0), class_names=("person", "dog"))
    _run_single(block, _make_frame(frame_number=1), class_names=("person", "dog"))
    # Fake adds a second "person" instance on the session's third frame.
    result = _run_single(
        block, _make_frame(frame_number=2), class_names=("person", "dog")
    )
    predictions = result[0]["predictions"]

    assert predictions.tracker_id.tolist() == [0, 1, 2]
    assert list(predictions.data["class_name"]) == ["person", "dog", "person"]
    assert predictions.class_id.tolist() == [0, 1, 0]


def test_threshold_filters_low_score_objects():
    block, _ = _make_block_with_fake_model()

    # Scores are 0.9 ("person") and 0.8 ("dog"); cut between them.
    result = _run_single(
        block, _make_frame(), class_names=("person", "dog"), threshold=0.85
    )
    predictions = result[0]["predictions"]

    assert len(predictions) == 1
    assert list(predictions.data["class_name"]) == ["person"]


def test_comma_separated_class_names_string_is_accepted():
    block, fake = _make_block_with_fake_model()

    block.run(
        images=[_make_frame()],
        class_names="person, dog",
        model_id="sam3video",
        threshold=0.0,
    )

    assert fake.calls[0][1]["text"] == ["person", "dog"]


# ---------------------------------------------------------------------------
# Visual tracking
# ---------------------------------------------------------------------------


def test_visual_points_prompt_once_then_track():
    block, fake = _make_visual_block_with_fake_model()
    points = [
        {"x": 10, "y": 12, "positive": True},
        {"x": 14, "y": 16, "positive": False},
    ]

    results = []
    for frame_number in range(3):
        results.append(
            block.run(
                images=[_make_frame(frame_number=frame_number)],
                class_names=None,
                model_id="sam3video",
                threshold=0.0,
                tracking_mode="visual",
                points=points,
            )
        )

    assert [call[0] for call in fake.calls] == ["prompt", "track", "track"]
    assert fake.calls[0][1]["points"] == [
        (10.0, 12.0, True),
        (14.0, 16.0, False),
    ]
    for result in results:
        predictions = result[0]["predictions"]
        assert len(predictions) == 1
        assert predictions.tracker_id.tolist() == [0]
        assert predictions.data["class_name"].tolist() == ["foreground"]


def test_visual_boxes_and_points_keep_prompt_metadata():
    block, fake = _make_visual_block_with_fake_model()

    result = block.run(
        images=[_make_frame()],
        class_names=None,
        model_id="sam3trackervideo",
        threshold=0.0,
        tracking_mode="visual",
        points=[{"x": 14, "y": 16, "positive": True}],
        boxes=[_make_box_detections()],
    )

    predictions = result[0]["predictions"]
    assert fake.calls[0][1]["bboxes"] == [(10.0, 20.0, 80.0, 90.0)]
    assert len(predictions) == 2
    assert predictions.tracker_id.tolist() == [0, 1]
    assert predictions.class_id.tolist() == [7, 0]
    assert predictions.confidence.tolist() == pytest.approx([0.91, 1.0])
    assert predictions.data["class_name"].tolist() == ["vehicle", "foreground"]


def test_visual_every_n_frames_mode_reprompts_after_interval():
    block, fake = _make_visual_block_with_fake_model()

    for frame_number in range(5):
        block.run(
            images=[_make_frame(frame_number=frame_number)],
            class_names=None,
            model_id="sam3trackervideo",
            threshold=0.0,
            tracking_mode="visual",
            points=[{"x": 10, "y": 12, "positive": True}],
            prompt_mode="every_n_frames",
            prompt_interval=2,
        )

    assert [call[0] for call in fake.calls] == [
        "prompt",
        "track",
        "track",
        "prompt",
        "track",
    ]


def test_tensor_visual_mode_emits_native_predictions():
    block = TensorSegmentAnything3VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    fake = _FakeVisualModel()
    block._model = fake
    block._current_model_id = "sam3trackervideo"

    result = block.run(
        images=[_make_frame()],
        class_names=None,
        model_id="sam3video",
        threshold=0.0,
        tracking_mode="visual",
        points=[{"x": 10, "y": 12, "positive": True}],
    )

    predictions = result[0]["predictions"]
    assert len(predictions) == 1
    assert predictions.bboxes_metadata[0]["tracker_id"] == 0
