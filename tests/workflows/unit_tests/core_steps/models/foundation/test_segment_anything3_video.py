"""Unit tests for the SAM3 Video Tracker workflow block."""

import sys
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import MagicMock, call

import numpy as np
import pytest
import supervision as sv
import torch

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3_video import (
    v1 as sam3_video_module,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_video import (
    v1_tensor as sam3_video_tensor_module,
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
from inference_models.models.base.object_detection import Detections

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
        self.calls.append(
            (
                "prompt",
                {"text": list(text), "first_pixel": image[0, 0].tolist()},
            )
        )
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
                    "first_pixel": image[0, 0].tolist(),
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
    assert manifest.pvs_model_id == "sam3trackervideo"
    assert manifest.class_names == ["person", "forklift"]
    assert manifest.threshold == 0.5


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_schema_keeps_mode_and_prompts_in_primary_fields(manifest_type):
    properties = manifest_type.model_json_schema()["properties"]

    assert properties["tracking_mode"]["always_visible"] is True
    assert properties["class_names"]["relevant_for"]["tracking_mode"] == {
        "values": ["concept"],
        "required": True,
    }
    for prompt_name in ("points", "boxes"):
        assert properties[prompt_name]["relevant_for"]["tracking_mode"] == {
            "values": ["visual"],
            "required": True,
        }


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_schema_shows_mode_specific_model_id(manifest_type):
    properties = manifest_type.model_json_schema()["properties"]

    assert properties["model_id"]["title"] == "Model Id"
    assert properties["model_id"]["default"] == "sam3video"
    assert properties["model_id"]["relevant_for"] == {
        "tracking_mode": {"values": ["concept"]}
    }
    assert properties["pvs_model_id"]["title"] == "Model Id"
    assert properties["pvs_model_id"]["default"] == "sam3trackervideo"
    assert properties["pvs_model_id"]["relevant_for"] == {
        "tracking_mode": {"values": ["visual"]}
    }


def test_manifest_requires_class_names():
    with pytest.raises(Exception):
        BlockManifest.model_validate(
            {
                "type": "roboflow_core/sam3_video@v1",
                "name": "sam3_video_step",
                "images": "$inputs.image",
            }
        )


def test_manifest_accepts_empty_class_names_for_backward_compatibility():
    data = {
        "type": "roboflow_core/sam3_video@v1",
        "name": "sam3_video_step",
        "images": "$inputs.image",
        "class_names": [],
    }
    manifest = BlockManifest.model_validate(data)
    tensor_manifest = TensorBlockManifest.model_validate(data)

    assert manifest.class_names == []
    assert tensor_manifest.class_names == []


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
    assert manifest.model_id == "sam3video"
    assert manifest.pvs_model_id == "sam3trackervideo"


def test_manifest_keeps_mode_specific_model_overrides():
    manifest = BlockManifest.model_validate(
        {
            "type": "roboflow_core/sam3_video@v1",
            "name": "sam3_video_step",
            "images": "$inputs.image",
            "tracking_mode": "visual",
            "boxes": "$steps.detector.predictions",
            "model_id": "custom-concept-model",
            "pvs_model_id": "custom-visual-model",
        }
    )

    assert manifest.model_id == "custom-concept-model"
    assert manifest.pvs_model_id == "custom-visual-model"


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

    assert manifest.model_id == "sam3video"
    assert manifest.pvs_model_id == "sam3trackervideo"


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


@pytest.mark.parametrize(
    "block_type", [SegmentAnything3VideoBlockV1, TensorSegmentAnything3VideoBlockV1]
)
def test_run_selects_model_id_for_tracking_mode(block_type):
    block = block_type(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    block._get_model = MagicMock(return_value=object())

    block.run(
        images=[],
        class_names=[],
        model_id="custom-concept-model",
        pvs_model_id="custom-visual-model",
        threshold=0.5,
    )
    block.run(
        images=[],
        class_names=None,
        model_id="custom-concept-model",
        pvs_model_id="custom-visual-model",
        threshold=0.5,
        tracking_mode="visual",
        points=[{"x": 10, "y": 20, "positive": True}],
    )

    assert block._get_model.call_args_list == [
        call(model_id="custom-concept-model"),
        call(model_id="custom-visual-model"),
    ]


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


def test_numpy_concept_and_visual_modes_convert_bgr_frames_to_rgb():
    concept_block, concept_model = _make_block_with_fake_model()
    visual_block, visual_model = _make_visual_block_with_fake_model()
    concept_frame = _make_frame(shape=(4, 4, 3))
    visual_frame = _make_frame(shape=(4, 4, 3))
    concept_frame.numpy_image[0, 0] = [10, 20, 30]
    visual_frame.numpy_image[0, 0] = [40, 50, 60]

    _run_single(concept_block, concept_frame)
    visual_block.run(
        images=[visual_frame],
        class_names=None,
        model_id="sam3video",
        pvs_model_id="sam3trackervideo",
        threshold=0.0,
        tracking_mode="visual",
        points=[{"x": 1, "y": 1, "positive": True}],
    )

    assert concept_model.calls[0][1]["first_pixel"] == [30, 20, 10]
    assert visual_model.calls[0][1]["first_pixel"] == [60, 50, 40]


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
                pvs_model_id="sam3trackervideo",
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
        model_id="sam3video",
        pvs_model_id="sam3trackervideo",
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
            model_id="sam3video",
            pvs_model_id="sam3trackervideo",
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
    assert [
        call[1]["had_prior_state"] for call in fake.calls if call[0] == "prompt"
    ] == [False, False]


def test_tensor_visual_empty_path_does_not_prepare_model_frame(monkeypatch):
    block = TensorSegmentAnything3VideoBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    fake = _FakeVisualModel()
    block._model = fake
    block._current_model_id = "sam3trackervideo"
    frame_for_model = MagicMock(side_effect=AssertionError("frame was prepared"))
    monkeypatch.setattr(
        sam3_video_tensor_module,
        "_frame_for_model",
        frame_for_model,
    )

    result = block.run(
        images=[_make_frame()],
        class_names=None,
        model_id="sam3video",
        pvs_model_id="sam3trackervideo",
        threshold=0.0,
        tracking_mode="visual",
        points=None,
        boxes=None,
    )

    assert len(result[0]["predictions"]) == 0
    frame_for_model.assert_not_called()


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
        pvs_model_id="sam3trackervideo",
        threshold=0.0,
        tracking_mode="visual",
        points=[{"x": 10, "y": 12, "positive": True}],
        boxes=[
            Detections(
                xyxy=torch.tensor([[10, 20, 80, 90]], dtype=torch.float32),
                class_id=torch.tensor([7], dtype=torch.int64),
                confidence=torch.tensor([0.91], dtype=torch.float32),
                bboxes_metadata=[{"class": "vehicle", "detection_id": "det-0"}],
            )
        ],
    )

    predictions = result[0]["predictions"]
    assert len(predictions) == 2
    assert predictions.bboxes_metadata[0]["tracker_id"] == 0
    assert predictions.bboxes_metadata[0]["class"] == "vehicle"
    assert predictions.bboxes_metadata[1]["tracker_id"] == 1
