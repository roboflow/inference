"""Unit tests for the SAM3 Video Tracker workflow block.

The block fetches the actual model via ``inference_models.AutoModel``;
we inject a fake model on the instance so tests don't touch
``inference_models`` or the real SAM3 weights.  The fake mirrors
``SAM3Video``'s concept-tracker contract: ``prompt(text=...)`` seeds a
session, ``track`` propagates it, and both return per-frame results
with detection scores and the prompt→object-ids mapping (which may grow
mid-stream as new matching objects enter the scene).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3_video.v1 import (
    BlockManifest,
    SegmentAnything3VideoBlockV1,
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
