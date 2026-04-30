"""Unit tests for the shared helpers in ``_streaming_video_common``."""

import numpy as np
import supervision as sv

from inference.core.workflows.core_steps.models.foundation._streaming_video_common import (
    BoxPromptMetadata,
    VideoSessionBookkeeping,
    build_obj_id_metadata_from_boxes,
    build_obj_id_metadata_from_text,
    decide_prompt_vs_track,
    extract_box_prompts,
    normalise_class_names,
)


def test_extract_box_prompts_empty_inputs_return_empty_lists():
    assert extract_box_prompts(None) == ([], [])
    empty = sv.Detections.empty()
    assert extract_box_prompts(empty) == ([], [])


def test_extract_box_prompts_returns_metadata_in_order():
    dets = sv.Detections(
        xyxy=np.array([[10, 20, 100, 200], [50, 60, 150, 160]], dtype=np.float32),
        confidence=np.array([0.9, 0.8], dtype=np.float32),
        class_id=np.array([0, 1], dtype=int),
    )
    dets.data["class_name"] = np.array(["person", "car"], dtype=object)
    dets.data["detection_id"] = np.array(["p0", "c0"], dtype=object)

    boxes, metas = extract_box_prompts(dets)

    assert boxes == [(10.0, 20.0, 100.0, 200.0), (50.0, 60.0, 150.0, 160.0)]
    assert metas[0].class_name == "person"
    assert metas[0].parent_id == "p0"
    assert metas[1].class_name == "car"
    assert metas[1].class_id == 1


def test_decide_prompt_vs_track_first_frame_prompts_when_fresh_with_prompts():
    session = VideoSessionBookkeeping()
    reset, should_prompt = decide_prompt_vs_track(
        session=session,
        frame_number=0,
        prompt_mode="first_frame",
        prompt_interval=30,
        has_prompts=True,
    )
    assert reset is True
    assert should_prompt is True


def test_decide_prompt_vs_track_first_frame_does_not_prompt_on_propagate():
    session = VideoSessionBookkeeping(
        state_dict={"marker": "alive"},
        last_frame_number=5,
    )
    reset, should_prompt = decide_prompt_vs_track(
        session=session,
        frame_number=6,
        prompt_mode="first_frame",
        prompt_interval=30,
        has_prompts=True,
    )
    assert reset is False
    assert should_prompt is False


def test_decide_prompt_vs_track_frame_rollback_resets():
    session = VideoSessionBookkeeping(
        state_dict={"marker": "alive"},
        last_frame_number=10,
    )
    reset, should_prompt = decide_prompt_vs_track(
        session=session,
        frame_number=0,
        prompt_mode="first_frame",
        prompt_interval=30,
        has_prompts=True,
    )
    assert reset is True
    assert should_prompt is True


def test_decide_prompt_vs_track_every_n_frames_honours_interval():
    session = VideoSessionBookkeeping(
        state_dict={"marker": "alive"},
        last_frame_number=3,
        frames_since_prompt=3,
    )
    reset, should_prompt = decide_prompt_vs_track(
        session=session,
        frame_number=4,
        prompt_mode="every_n_frames",
        prompt_interval=3,
        has_prompts=True,
    )
    assert reset is False
    assert should_prompt is True


def test_decide_prompt_vs_track_every_frame_requires_prompts():
    session = VideoSessionBookkeeping(
        state_dict={"marker": "alive"},
        last_frame_number=1,
    )
    reset, should_prompt = decide_prompt_vs_track(
        session=session,
        frame_number=2,
        prompt_mode="every_frame",
        prompt_interval=30,
        has_prompts=False,
    )
    assert reset is False
    assert should_prompt is False


def test_build_obj_id_metadata_from_boxes_zips_parallel_lists():
    obj_ids = np.array([5, 6], dtype=np.int64)
    metas = [
        BoxPromptMetadata(
            class_id=0, class_name="person", confidence=0.9, parent_id="p"
        ),
        BoxPromptMetadata(class_id=1, class_name="car", confidence=0.8, parent_id="c"),
    ]
    mapping = build_obj_id_metadata_from_boxes(obj_ids, metas)
    assert mapping[5].class_name == "person"
    assert mapping[6].class_name == "car"


def test_build_obj_id_metadata_from_text_uses_label_when_single():
    mapping = build_obj_id_metadata_from_text(
        obj_ids=np.array([0, 1], dtype=np.int64),
        class_names=["person"],
    )
    assert mapping[0].class_name == "person"
    assert mapping[1].class_name == "person"


def test_build_obj_id_metadata_from_text_falls_back_for_multiple_classes():
    mapping = build_obj_id_metadata_from_text(
        obj_ids=np.array([0, 1], dtype=np.int64),
        class_names=["person", "car"],
    )
    assert mapping[0].class_name == "foreground"


def test_normalise_class_names_variants():
    assert normalise_class_names(None) == []
    assert normalise_class_names("") == []
    assert normalise_class_names([]) == []
    assert normalise_class_names(["a", "b"]) == ["a", "b"]
    assert normalise_class_names("a, b ,  c") == ["a", "b", "c"]
