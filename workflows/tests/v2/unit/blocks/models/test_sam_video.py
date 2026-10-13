"""Contract tests, including compiled multi-run workflows, without real models.

The doubles copy the pinned HFStreamingVideoBase and SAM3Video signatures.
Visual results are tuples; concept results have SAM3VideoFrameResult's fields.
They deliberately mutate opaque handles in place and return a new wrapper on
each track. The indexed doubles run the pinned backend's prompt/track code
with transformers' streaming frame-index rules. These tests qualify
adapter/lifecycle contracts, not inference.
"""

import gc
import weakref
from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from inference_models.models.base.object_detection import Detections
from inference_models.models.sam2_video.sam2_video_hf import SAM2Video
from inference_models.models.sam3_tracker_video.sam3_tracker_video_hf import (
    SAM3TrackerVideo,
)
from pydantic import ValidationError
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.image.dynamic_crop import DynamicCropBlock
from roboflow_workflows.v2.blocks.models.cache import ModelCache, ModelRequest
from roboflow_workflows.v2.blocks.models.detection import ObjectDetectionBlock
from roboflow_workflows.v2.blocks.models.sam_video import (
    SAM2_VIDEO_MODEL_ID,
    SAM3_CONCEPT_MODEL_ID,
    SAM3_VISUAL_MODEL_ID,
    SegmentAnything2VideoBlock,
    SegmentAnything3VideoBlock,
    VideoSourceCapacityError,
)
from roboflow_workflows.v2.data.batch import (
    Batch,
    EntryMetadata,
    InputValue,
    SampleContext,
)
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.masks.grid import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    SegmentationView,
)
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAMES_KEY
from roboflow_workflows.v2.data.predictions import NATIVE_KINDS
from roboflow_workflows.v2.data.row_identity import row_record
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import StepExecutionError
from roboflow_workflows.v2.engine.plan import CompileOptions
from roboflow_workflows.v2.engine.reactions.testing import block_call
from tests.v2.support.models.fake_task_models import NmsDetector

SIZE = (8, 10)
POINTS = [{"x": 3, "y": 4, "positive": True}, [5, 4, False]]


class _VisualBackend:
    def __init__(self):
        self.calls = []
        self.handles = []
        self.fail = False
        self.bad = None

    def prompt(
        self,
        image,
        bboxes=None,
        text=None,
        state_dict=None,
        clear_old_prompts=True,
        frame_idx=0,
        points=None,
    ):
        assert text is None and state_dict is None and clear_old_prompts
        assert image.ndim == 3 and image.shape[2] == 3
        state = {
            "step": 0,
            "count": len(bboxes or []) + int(bool(points)),
            "token": object(),
        }
        handle = {"opaque": state}
        self.handles.append(handle)
        self.calls.append(("prompt", handle, bboxes, points, frame_idx))
        result = self._result(image, handle)

        return result

    def track(self, image, state_dict=None):
        assert state_dict is not None
        state_dict["opaque"]["step"] += 1
        self.calls.append(("track", state_dict))
        if self.fail:
            raise RuntimeError("backend mutated then failed")
        result = self._result(image, {"opaque": state_dict["opaque"]})

        return result

    def _result(self, image, handle):
        state = handle["opaque"]
        count = state["count"]
        masks = np.zeros((count, *image.shape[:2]), dtype=bool)
        for i in range(count):
            masks[i, 1:4, 1 + i : 4 + i] = True
        ids = np.arange(count, dtype=np.int64)
        result = (masks, ids, handle)
        if self.bad is not None:
            result = self.bad(result)

        return result


@dataclass
class _ConceptRecord:
    masks: np.ndarray
    object_ids: np.ndarray
    scores: np.ndarray
    boxes: np.ndarray
    prompt_to_object_ids: dict
    state_dict: dict


class _ConceptBackend:
    def __init__(self):
        self.calls = []
        self.bad = None

    def prompt(self, image, text, state_dict=None, clear_old_prompts=True):
        assert state_dict is None and clear_old_prompts
        handle = {"opaque": {"step": 0, "names": list(text), "token": object()}}
        self.calls.append(("prompt", handle, text))
        result = self._result(image, handle)

        return result

    def track(self, image, state_dict=None):
        state_dict["opaque"]["step"] += 1
        self.calls.append(("track", state_dict))
        result = self._result(image, {"opaque": state_dict["opaque"]})

        return result

    def _result(self, image, handle):
        state = handle["opaque"]
        count = 1 if state["step"] == 0 else 2
        masks = np.zeros((count, *image.shape[:2]), dtype=bool)
        masks[:, 2:5, 3:6] = True
        ids = np.array([7, 42][:count], dtype=np.int64)
        names = state["names"]
        mapping = {names[0]: [7]}
        if count == 2:
            mapping.setdefault(names[-1], []).append(42)
        result = _ConceptRecord(
            masks=masks,
            object_ids=ids,
            scores=np.array([0.9, 0.6][:count], dtype=np.float32),
            boxes=np.array([[3, 2, 5, 4]] * count, dtype=np.float32),
            prompt_to_object_ids=mapping,
            state_dict=handle,
        )
        if self.bad is not None:
            result = self.bad(result)

        return result


def _image(source="a", frame=0, *, size=SIZE, video=True):
    metadata = (
        VideoMetadata(
            video_identifier=source,
            frame_number=frame,
            frame_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        if video
        else None
    )
    image = ImageData.from_tensor(
        torch.zeros((3, *size), dtype=torch.uint8),
        image_id=f"{source}:{frame}",
        video_metadata=metadata,
    )

    return image


def _boxes(image, *, empty=False):
    prediction = Detections(
        xyxy=torch.zeros((0, 4)) if empty else torch.tensor([[1.0, 1.0, 5.0, 6.0]]),
        class_id=torch.zeros(0, dtype=torch.int64) if empty else torch.tensor([2]),
        confidence=torch.zeros(0) if empty else torch.tensor([0.8]),
        image_metadata={
            **FrameRef.from_image(image).prediction_metadata(),
            CLASS_NAMES_KEY: {2: "car"},
        },
        bboxes_metadata=[] if empty else [{"upstream": "box-row"}],
    )

    return prediction


def _direct(block, image, *, wanted=None, **kwargs):
    with block_call(block, indices=((0,),), wanted=wanted):
        result = block.run(Batch([image]), **kwargs)[0]

    return result


def _plan(
    *,
    sam3=False,
    detector=False,
    crop_from=None,
    mode="run",
    outputs=("predictions", "segmentation"),
    **params,
):
    block = SegmentAnything3VideoBlock if sam3 else SegmentAnything2VideoBlock
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]},
            *(
                {"type": "WorkflowParameter", "name": name}
                for name in ("model", "mode", "names")
            ),
        ],
        "steps": [
            {"type": block.type, "name": "sam", "image": "$inputs.image", **params}
        ],
        "outputs": [
            {"type": "JsonField", "name": name, "selector": f"$steps.sam.{name}"}
            for name in outputs
        ],
    }
    if detector:
        definition["steps"].insert(
            0,
            {
                "type": ObjectDetectionBlock.type,
                "name": "detector",
                "image": "$inputs.image",
                "model_id": "detector",
                "confidence": 0.0,
            },
        )
        definition["steps"][1]["boxes"] = "$steps.detector.predictions"
    if crop_from is not None:
        definition["steps"].append(
            {
                "type": DynamicCropBlock.type,
                "name": "crop",
                "image": "$inputs.image",
                "predictions": f"$steps.sam.{crop_from}",
            }
        )
        definition["outputs"].append(
            {"type": "JsonField", "name": "crops", "selector": "$steps.crop.crops"}
        )
    catalogue = Catalogue(
        [
            DynamicCropBlock,
            ObjectDetectionBlock,
            SegmentAnything2VideoBlock,
            SegmentAnything3VideoBlock,
        ],
        kinds=[*NATIVE_KINDS, INSTANCE_SEGMENTATION_VIEW_KIND],
    )
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )

    return plan


@pytest.mark.parametrize("execution", ["run", "phases"])
def test_compiled_interleaved_sources_retain_distinct_handles_and_fresh_rows(execution):
    backend = _VisualBackend()
    cache = ModelCache(supplied={SAM2_VIDEO_MODEL_ID: backend})
    session = _plan(mode=execution, points=POINTS).create_session({"models": cache})
    first = session.run({"image": [_image("a", 0), _image("b", 0)]}).rows()
    second = session.run({"image": [_image("b", 1), _image("a", 1)]}).rows()
    third = session.run({"image": [_image("a", 2), _image("b", 2)]}).rows()
    a, b = backend.handles
    assert a["opaque"] is not b["opaque"]
    assert backend.calls[2][1] is b and backend.calls[3][1] is a
    assert (
        backend.calls[4][1] is not a
    )  # the most recently returned wrapper is retained
    assert backend.calls[4][1]["opaque"] is a["opaque"]
    assert a["opaque"]["step"] == b["opaque"]["step"] == 2
    for row in [*first, *second, *third]:
        pred = row["predictions"]
        assert pred.bboxes_metadata == [{"tracker_id": 0}]
        assert isinstance(row["segmentation"], SegmentationView)
        assert pred.xyxy.tolist() == [[1.0, 1.0, 4.0, 4.0]]
        assert row["segmentation"].detections.image_metadata == pred.image_metadata
    assert (
        len(
            {
                row_record(row["predictions"]).namespace
                for row in [*first, *second, *third]
            }
        )
        == 6
    )
    assert frame_of(second[0]["predictions"]).image_id == "b:1"


@pytest.mark.parametrize("execution", ["run", "phases"])
def test_compiled_reset_new_session_and_model_switch_drop_handles(execution):
    backend = _VisualBackend()
    cache = ModelCache(supplied={"one": backend, "two": backend})
    plan = _plan(mode=execution, points=POINTS, model_id="$inputs.model")
    session = plan.create_session({"models": cache})
    for frame, model in enumerate(["one", "one", "two", "one"]):
        session.run({"image": [_image(frame=frame)], "model": model})
    assert [call[0] for call in backend.calls] == [
        "prompt",
        "track",
        "prompt",
        "prompt",
    ]
    session.update(plan, reset=True)
    session.run({"image": [_image(frame=4)], "model": "one"})
    other = plan.create_session({"models": cache})
    other.run({"image": [_image(frame=5)], "model": "one"})
    assert len(backend.handles) == 5
    assert len({id(h["opaque"]) for h in backend.handles}) == 5


@pytest.mark.parametrize("execution", ["run", "phases"])
def test_compiled_mode_and_concept_changes_start_fresh_sessions(execution):
    visual, concept = _VisualBackend(), _ConceptBackend()
    plan = _plan(
        sam3=True,
        mode=execution,
        tracking_mode="$inputs.mode",
        class_names="$inputs.names",
        points=POINTS,
    )
    session = plan.create_session(
        {"models": {SAM3_VISUAL_MODEL_ID: visual, SAM3_CONCEPT_MODEL_ID: concept}}
    )
    options = [
        ("concept", ["car"]),
        ("concept", ["car"]),
        ("visual", []),
        ("concept", ["car"]),
        ("concept", ["dog"]),
    ]
    for frame, (mode, names) in enumerate(options):
        session.run({"image": [_image(frame=frame)], "mode": mode, "names": names})
    assert [c[0] for c in concept.calls] == ["prompt", "track", "prompt", "prompt"]
    assert len(visual.handles) == 1


@pytest.mark.parametrize(
    "prompt_mode,interval,expected",
    [
        ("first_frame", 1, ["prompt", "track", "track", "track"]),
        ("every_frame", 30, ["prompt"] * 4),
        ("every_n_frames", 1, ["prompt", "track", "prompt", "track"]),
        ("every_n_frames", 2, ["prompt", "track", "track", "prompt"]),
    ],
)
def test_compiled_visual_prompt_policies(prompt_mode, interval, expected):
    backend = _VisualBackend()
    session = _plan(
        points=POINTS, prompt_mode=prompt_mode, prompt_interval=interval
    ).create_session({"models": {SAM2_VIDEO_MODEL_ID: backend}})
    for frame in range(4):
        session.run({"image": [_image(frame=frame)]})
    assert [c[0] for c in backend.calls] == expected
    assert all(
        call[1]["opaque"]["step"] <= interval or prompt_mode == "first_frame"
        for call in backend.calls
    )


def test_empty_first_frame_delays_prompt_and_due_empty_prompts_keep_tracking():
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    first = _direct(block, _image(), points=[])["predictions"]
    assert len(first) == 0 and tuple(first.mask.shape) == (0, *SIZE)
    assert not backend.calls
    _direct(block, _image(frame=1), points=POINTS, prompt_mode="every_frame")
    _direct(block, _image(frame=2), points=[], prompt_mode="every_frame")
    assert [c[0] for c in backend.calls] == ["prompt", "track"]


def test_box_and_point_prompts_preserve_metadata_and_use_noncolliding_classes():
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    image = _image(frame=19)
    output = _direct(block, image, boxes=Batch([_boxes(image)]), points=POINTS)[
        "predictions"
    ]
    # Source frame 19 is the fresh backend session's local frame 0.
    assert backend.calls[0][2:] == (
        [(1.0, 1.0, 5.0, 6.0)],
        [(3.0, 4.0, True), (5.0, 4.0, False)],
        0,
    )
    assert output.class_id.tolist() == [2, 3]
    assert output.image_metadata[CLASS_NAMES_KEY] == {2: "car", 3: "foreground"}
    assert output.bboxes_metadata == [
        {"prompt_metadata": {"upstream": "box-row"}, "tracker_id": 0},
        {"tracker_id": 1},
    ]
    filtered = _direct(block, _image(frame=20), confidence=0.9)["predictions"]
    assert filtered.class_id.tolist() == [3] and filtered.bboxes_metadata == [
        {"tracker_id": 1}
    ]


def test_frame_rollback_size_change_direct_reset_and_capacity():
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    for frame, size in [(5, SIZE), (6, SIZE), (2, SIZE), (3, (9, 12))]:
        _direct(block, _image(frame=frame, size=size), points=POINTS)
    assert [c[0] for c in backend.calls] == ["prompt", "track", "prompt", "prompt"]
    with pytest.raises(VideoSourceCapacityError):
        _direct(block, _image("b"), points=POINTS, max_sources=1)
    block.reset_state()
    _direct(block, _image("b"), points=POINTS, max_sources=1)
    assert backend.calls[-1][0] == "prompt"


def test_concept_mapping_rebuilt_for_new_objects_and_scores_filter_per_frame():
    backend = _ConceptBackend()
    block = SegmentAnything3VideoBlock({SAM3_CONCEPT_MODEL_ID: backend})
    first = _direct(block, _image(), class_names="car, dog")["predictions"]
    second = _direct(block, _image(frame=1), class_names=["car", "dog"])["predictions"]
    assert first.class_id.tolist() == [0]
    assert second.class_id.tolist() == [0, 1]
    assert second.bboxes_metadata == [{"tracker_id": 7}, {"tracker_id": 42}]
    assert second.xyxy.tolist() == [[3.0, 2.0, 5.0, 4.0]] * 2
    assert second.confidence.tolist() == pytest.approx([0.9, 0.6])
    assert row_record(first).namespace != row_record(second).namespace
    filtered = _direct(
        block, _image(frame=2), class_names=["car", "dog"], confidence=0.7
    )["predictions"]
    assert len(filtered) == 1
    empty = _direct(block, _image(frame=3), class_names=[])["predictions"]
    assert len(empty) == 0 and len(backend.calls) == 3


def test_unknown_concept_object_gets_truthful_foreground_class():
    backend = _ConceptBackend()
    backend.bad = lambda r: _ConceptRecord(
        r.masks, r.object_ids, r.scores, r.boxes, {}, r.state_dict
    )
    output = _direct(
        SegmentAnything3VideoBlock({SAM3_CONCEPT_MODEL_ID: backend}),
        _image(),
        class_names=["car"],
    )["predictions"]
    assert output.class_id.tolist() == [1]
    assert output.image_metadata[CLASS_NAMES_KEY][1] == "foreground"


@pytest.mark.parametrize(
    "bad,match",
    [
        (lambda r: (r[0], r[1]), "visual backend"),
        (lambda r: (r[0][:, :2], r[1], r[2]), "mask size"),
        (lambda r: (r[0].astype(float), r[1], r[2]), "binary bool"),
        (lambda r: (r[0], r[1].astype(float), r[2]), "integer array"),
        (lambda r: (r[0], np.array([0, 0]), r[2]), "unique"),
        (lambda r: (r[0], r[1], None), "opaque state_dict"),
    ],
)
def test_malformed_visual_result_drops_source_and_next_call_reprompts(bad, match):
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    _direct(block, _image(), points=POINTS)
    backend.bad = bad
    with pytest.raises(ValueError, match=match):
        _direct(block, _image(frame=1), points=POINTS)
    backend.bad = None
    _direct(block, _image(frame=2), points=POINTS)
    assert len(backend.handles) == 2


@pytest.mark.parametrize(
    "bad,match",
    [
        (lambda r: (r.masks, r.object_ids, r.state_dict), "concept backend"),
        (
            lambda r: _ConceptRecord(
                r.masks,
                r.object_ids,
                np.array([np.nan]),
                r.boxes,
                r.prompt_to_object_ids,
                r.state_dict,
            ),
            "scores",
        ),
        (
            lambda r: _ConceptRecord(
                r.masks,
                r.object_ids,
                r.scores,
                np.zeros((1, 3)),
                r.prompt_to_object_ids,
                r.state_dict,
            ),
            "boxes",
        ),
        (
            lambda r: _ConceptRecord(
                r.masks, r.object_ids, r.scores, r.boxes, {"wrong": [7]}, r.state_dict
            ),
            "registered prompts",
        ),
        (
            lambda r: _ConceptRecord(
                r.masks,
                r.object_ids,
                r.scores,
                r.boxes,
                {"car": [7], "dog": [7]},
                r.state_dict,
            ),
            "multiple prompts",
        ),
    ],
)
def test_malformed_concept_record_is_rejected(bad, match):
    backend = _ConceptBackend()
    backend.bad = bad
    with pytest.raises(ValueError, match=match):
        _direct(
            SegmentAnything3VideoBlock({SAM3_CONCEPT_MODEL_ID: backend}),
            _image(),
            class_names=["car", "dog"],
        )


def test_compiled_backend_failure_discards_only_failed_stream():
    backend = _VisualBackend()
    session = _plan(points=POINTS).create_session(
        {"models": {SAM2_VIDEO_MODEL_ID: backend}}
    )
    session.run({"image": [_image("a"), _image("b")]})
    a, b = backend.handles
    backend.fail = True
    with pytest.raises(StepExecutionError, match="backend mutated then failed"):
        session.run({"image": [_image("a", 1)]})
    assert a["opaque"]["step"] == 1
    backend.fail = False
    session.run({"image": [_image("a", 2), _image("b", 1)]})
    assert backend.calls[-2][0] == "prompt" and backend.calls[-1][1] is b


def test_prompt_and_frame_validation_before_backend_call():
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    image = _image()
    with pytest.raises(ValueError, match="boxes were predicted"):
        _direct(block, image, boxes=Batch([_boxes(_image("other"))]))
    with pytest.raises(ValueError, match="outside image"):
        _direct(block, image, points=[[100, 1]])
    with pytest.raises(ValueError, match="align"):
        _direct(block, image, boxes=Batch([], indices=[]))
    bad = _boxes(image)
    bad.xyxy[0, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        _direct(block, image, boxes=Batch([bad]))
    assert not backend.calls
    with pytest.raises(ValidationError):
        SegmentAnything2VideoBlock.Params(image="$inputs.image", prompt_interval=0)
    with pytest.raises(ValueError, match="unique"):
        _direct(SegmentAnything3VideoBlock({}), image, class_names=["car", "car"])


def test_backend_empty_mask_sentinel_and_empty_masks_are_supported():
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    _direct(block, _image(), points=POINTS)
    backend.bad = lambda r: (
        np.zeros((0, 0, 0), dtype=bool),
        np.zeros(0, dtype=np.int64),
        r[2],
    )
    output = _direct(block, _image(frame=1))["predictions"]
    assert tuple(output.mask.shape) == (0, *SIZE)


@pytest.mark.parametrize("wanted", ["predictions", "segmentation"])
def test_output_demand_emits_only_requested_prediction_form(wanted):
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    output = _direct(block, _image(), points=POINTS, wanted={wanted})
    other = "segmentation" if wanted == "predictions" else "predictions"
    assert wanted in output and other not in output
    session = _plan(points=POINTS, outputs=(wanted,)).create_session(
        {"models": {SAM2_VIDEO_MODEL_ID: backend}}
    )
    compiled = session.run({"image": [_image()]}).rows()[0]
    assert list(compiled) == [wanted]


def test_compiled_source_context_fallback_is_per_member_and_video_metadata_wins():
    backend = _VisualBackend()
    session = _plan(points=POINTS).create_session(
        {"models": {SAM2_VIDEO_MODEL_ID: backend}}
    )
    for frame in range(2):
        images = [
            _image("same", frame, video=False),
            _image("same", frame, video=False),
        ]
        value = InputValue(
            images,
            metadata=EntryMetadata(
                sample={(0,): SampleContext("north"), (1,): SampleContext("south")}
            ),
        )
        session.run({"image": value})
    assert len(backend.handles) == 2
    assert backend.calls[2][1] is backend.handles[0]
    assert backend.calls[3][1] is backend.handles[1]


def test_request_package_change_starts_fresh_handle_without_model_reload():
    backend = _VisualBackend()
    cache = ModelCache(
        supplied={ModelRequest("sam", "a"): backend, ModelRequest("sam", "b"): backend}
    )
    block = SegmentAnything2VideoBlock(cache)
    for frame, package in enumerate(["a", "a", "b"]):
        _direct(
            block,
            _image(frame=frame),
            points=POINTS,
            model_id="sam",
            model_package_id=package,
        )
    assert [c[0] for c in backend.calls] == ["prompt", "track", "prompt"]


@pytest.mark.parametrize("execution", ["run", "phases"])
def test_compiled_detector_empty_first_frame_then_boxes_seed_and_track(execution):
    backend = _VisualBackend()
    cache = ModelCache(
        supplied={SAM2_VIDEO_MODEL_ID: backend, "detector": NmsDetector()}
    )
    session = _plan(detector=True, mode=execution).create_session({"models": cache})
    first = session.run({"image": [_image()]}).rows()[0]["predictions"]
    assert len(first) == 0 and not backend.calls
    image = _image(frame=1)
    image.tensor_image.fill_(100)
    seeded = session.run({"image": [image]}).rows()[0]["predictions"]
    assert len(seeded) == 2 and seeded.class_id.tolist() == [0, 1]
    tracked = session.run({"image": [_image(frame=2)]}).rows()[0]["predictions"]
    assert len(tracked) == 2
    assert [c[0] for c in backend.calls] == ["prompt", "track"]
    assert row_record(seeded).namespace != row_record(tracked).namespace


def test_transformed_prompt_rows_drop_reserved_host_mirrors():
    backend = _VisualBackend()
    image = _image()
    boxes = _boxes(image)
    original = {
        "_host_xyxy": [100, 200, 300, 400],
        "_host_class_id": 999,
        "_host_confidence": 0.1,
        "user": "preserved",
    }
    boxes.bboxes_metadata = [original]
    output = _direct(
        SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend}),
        image,
        boxes=Batch([boxes]),
    )["predictions"]
    assert output.bboxes_metadata == [
        {"tracker_id": 0, "prompt_metadata": {"user": "preserved"}}
    ]
    assert boxes.bboxes_metadata[0] is original and "_host_xyxy" in original


def test_visual_tracking_ignores_boxes_from_unread_frames():
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    _direct(block, _image(), points=POINTS)
    # V1 prompt policy ignores box contents on tracking-only frames.
    _direct(block, _image(frame=1), boxes=Batch([_boxes(_image("different"))]))
    assert backend.calls[-1][0] == "track"


def test_visual_result_must_have_one_id_per_prompt_and_only_prompted_ids():
    backend = _VisualBackend()
    backend.bad = lambda r: (np.concatenate([r[0], r[0]]), np.array([0, 1]), r[2])
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    with pytest.raises(ValueError, match="unprompted"):
        _direct(block, _image(), points=POINTS)
    backend.bad = None
    _direct(block, _image(), points=POINTS)
    backend.bad = lambda r: (r[0], np.array([44]), r[2])
    with pytest.raises(ValueError, match="unprompted"):
        _direct(block, _image(frame=1))


def test_compiled_repeated_source_members_are_processed_in_batch_order():
    backend = _VisualBackend()
    session = _plan(points=POINTS).create_session(
        {"models": {SAM2_VIDEO_MODEL_ID: backend}}
    )
    session.run({"image": [_image("a", 0), _image("b", 0), _image("a", 1)]})
    assert [c[0] for c in backend.calls] == ["prompt", "prompt", "track"]
    assert backend.calls[-1][1] is backend.handles[0]
    assert backend.handles[0]["opaque"]["step"] == 1


def test_video_source_metadata_takes_precedence_over_sample_context():
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    for frame in range(2):
        with block_call(block, indices=((0,),), source_id=f"different-{frame}"):
            block.run(Batch([_image("fixed-video", frame)]), points=POINTS)
    assert [c[0] for c in backend.calls] == ["prompt", "track"]


def test_seed_with_empty_backend_result_keeps_handle_for_later_visible_objects():
    backend = _VisualBackend()
    block = SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend})
    backend.bad = lambda r: (
        np.zeros((0, 0, 0), dtype=bool),
        np.zeros(0, dtype=np.int64),
        r[2],
    )
    first = _direct(block, _image(), points=POINTS)["predictions"]
    assert len(first) == 0
    backend.bad = None
    second = _direct(block, _image(frame=1))["predictions"]
    assert second.bboxes_metadata == [{"tracker_id": 0}]
    assert [c[0] for c in backend.calls] == ["prompt", "track"]
    assert backend.calls[1][1] is backend.handles[0]


@pytest.mark.parametrize(
    "field,value",
    [
        ("class_id", torch.tensor([2, 3])),
        ("confidence", torch.tensor([])),
        ("bboxes_metadata", []),
        ("class_id", torch.tensor([2.5])),
    ],
)
def test_malformed_prompt_rows_fail_before_backend(field, value):
    backend = _VisualBackend()
    image = _image()
    boxes = _boxes(image)
    setattr(boxes, field, value)
    with pytest.raises(ValueError, match="align|integers"):
        _direct(
            SegmentAnything2VideoBlock({SAM2_VIDEO_MODEL_ID: backend}),
            image,
            boxes=Batch([boxes]),
        )
    assert not backend.calls


# Explicit raster extents, including the last occupied row and column.
RASTER_CASES = [
    ([(3, 2)], [3.0, 2.0, 4.0, 3.0], (1, 1)),
    ([(x, 2) for x in range(1, 6)], [1.0, 2.0, 6.0, 3.0], (1, 5)),
    ([(3, y) for y in range(1, 7)], [3.0, 1.0, 4.0, 7.0], (6, 1)),
    ([(9, 7)], [9.0, 7.0, 10.0, 8.0], (1, 1)),
    ([(x, y) for y in range(2, 6) for x in range(3, 8)], [3.0, 2.0, 8.0, 6.0], (4, 5)),
    ([], [0.0, 0.0, 0.0, 0.0], None),
]


def _raster_mask(pixels):
    mask = torch.zeros(SIZE, dtype=torch.bool)
    for x, y in pixels:
        mask[y, x] = True
    return mask


@pytest.mark.parametrize("execution", ["run", "phases"])
@pytest.mark.parametrize("route", ["predictions", "segmentation"])
@pytest.mark.parametrize("pixels,box,crop_size", RASTER_CASES)
def test_compiled_video_raster_bounds_compose_with_dynamic_crop(
    execution, route, pixels, box, crop_size
):
    backend = _VisualBackend()
    mask = _raster_mask(pixels)
    backend.bad = lambda r: (mask[None].numpy(), r[1], r[2])
    plan = _plan(mode=execution, points=POINTS, crop_from=route)
    session = plan.create_session({"models": {SAM2_VIDEO_MODEL_ID: backend}})
    for frame in range(2):
        row = session.run({"image": [_image(frame=frame)]}).rows()[0]
        predictions = row["predictions"]
        if crop_size is None:
            assert len(predictions) == 0 and row["crops"] == []
        else:
            assert predictions.xyxy.tolist() == [box]
            assert len(row["crops"]) == 1
            assert row["crops"][0].size_hw == crop_size
            assert row["crops"][0].parent_row is not None
    assert [c[0] for c in backend.calls] == ["prompt", "track"]
