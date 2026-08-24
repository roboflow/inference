"""Tests for the stateful Video Classification Model workflow block."""

import importlib
import math
from datetime import datetime
from typing import Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import inference.core.env as core_env
from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_video_multi_label_classification_kind,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.serializers import (
    serialize_video_multi_label_classification_kind,
)
from inference.core.workflows.core_steps.models.roboflow.video_classification import (
    v1 as video_classification_module,
)
from inference.core.workflows.core_steps.models.roboflow.video_classification.v1 import (
    BlockManifest,
    VideoClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.video_classification.v1_tensor import (
    BlockManifest as TensorBlockManifest,
)
from inference.core.workflows.core_steps.models.roboflow.video_classification.v1_tensor import (
    VideoClassificationModelBlockV1 as TensorVideoClassificationModelBlockV1,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoIntervalClassification,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    VIDEO_MULTI_LABEL_CLASSIFICATION_PREDICTION_KIND,
)


class _FakeVideoClassificationModel:
    def __init__(self, responses: Optional[List[List[Dict]]] = None):
        self.responses = list(responses or [])
        self.calls = []

    def temporal_localization(
        self,
        frames,
        class_names,
        input_color_format="rgb",
        fps=None,
    ):
        recorded_frames = [
            frame.clone() if isinstance(frame, torch.Tensor) else frame.copy()
            for frame in frames
        ]
        self.calls.append(
            {
                "frames": recorded_frames,
                "class_names": list(class_names),
                "input_color_format": input_color_format,
                "fps": fps,
            }
        )
        if not self.responses:
            return []
        return self.responses.pop(0)


def _make_frame(
    frame_number: int,
    video_id: str = "stream-0",
    fps: Optional[float] = 4.0,
    measured_fps: Optional[float] = None,
    bgr_color: Optional[List[int]] = None,
    tensor_rgb_color: Optional[List[int]] = None,
) -> WorkflowImageData:
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        fps=fps,
        measured_fps=measured_fps,
        frame_timestamp=datetime(2024, 1, 1),
    )
    image_kwargs = {}
    if tensor_rgb_color is not None:
        image_kwargs["tensor_image"] = (
            torch.tensor(tensor_rgb_color, dtype=torch.uint8)
            .view(3, 1, 1)
            .expand(3, 2, 2)
            .clone()
        )
    else:
        color = bgr_color or [frame_number % 255, 10, 20]
        image_kwargs["numpy_image"] = np.full((2, 2, 3), color, dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"{video_id}:{frame_number}"),
        video_metadata=metadata,
        **image_kwargs,
    )


def _make_block(
    responses: Optional[List[List[Dict]]] = None,
    tensor: bool = False,
):
    block_type = (
        TensorVideoClassificationModelBlockV1
        if tensor
        else VideoClassificationModelBlockV1
    )
    block = block_type(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    model = _FakeVideoClassificationModel(responses=responses)
    block._model = model
    return block, model


def _run(
    block,
    frame: WorkflowImageData,
    class_names=("walk", "run"),
    window_size_seconds=1.0,
    sampling_fps=2.0,
):
    return block.run(
        images=[frame],
        class_names=list(class_names),
        model_id="cosmos-3-edge",
        window_size_seconds=window_size_seconds,
        sampling_fps=sampling_fps,
    )[0]


def _timeline_as_dicts(result):
    return [entry.model_dump() for entry in result["timeline"]]


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_parses_classes_and_declares_outputs(manifest_type):
    manifest = manifest_type.model_validate(
        {
            "type": "roboflow_core/video_classification_model@v1",
            "name": "video_classifier",
            "images": "$inputs.image",
            "class_names": "walk, run",
        }
    )

    assert manifest.class_names == "walk, run"
    assert manifest.model_id == "cosmos-3-edge"
    assert manifest.window_size_seconds == 2.0
    assert manifest.sampling_fps == 4.0
    assert manifest_type.get_parameters_accepting_batches() == ["images"]
    assert manifest_type.describe_outputs()[0].kind == [
        VIDEO_MULTI_LABEL_CLASSIFICATION_PREDICTION_KIND
    ]
    assert manifest_type.describe_outputs()[1].kind == [LIST_OF_VALUES_KIND]


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_requires_class_names(manifest_type):
    with pytest.raises(Exception):
        manifest_type.model_validate(
            {
                "type": "roboflow_core/video_classification_model@v1",
                "name": "video_classifier",
                "images": "$inputs.image",
            }
        )


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_size_seconds", 0),
        ("window_size_seconds", -1),
        ("window_size_seconds", math.inf),
        ("window_size_seconds", math.nan),
        ("sampling_fps", 0),
        ("sampling_fps", -1),
        ("sampling_fps", math.inf),
        ("sampling_fps", math.nan),
    ],
)
def test_manifest_rejects_non_positive_or_non_finite_window_inputs(
    manifest_type, field, value
):
    data = {
        "type": "roboflow_core/video_classification_model@v1",
        "name": "video_classifier",
        "images": "$inputs.image",
        "class_names": ["walk"],
        field: value,
    }

    with pytest.raises(Exception, match="positive and finite"):
        manifest_type.model_validate(data)


def test_window_waits_for_completion_and_sends_sampled_rgb_frames():
    block, model = _make_block()
    colors = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]

    for frame_number, color in enumerate(colors[:3]):
        result = _run(block, _make_frame(frame_number, bgr_color=color))
        assert result == {"timeline": [], "active_classes": []}
        assert model.calls == []

    result = _run(block, _make_frame(3, bgr_color=colors[3]))

    assert result == {"timeline": [], "active_classes": []}
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["class_names"] == ["walk", "run"]
    assert call["input_color_format"] == "rgb"
    assert call["fps"] == 2.0
    assert len(call["frames"]) == 2
    np.testing.assert_array_equal(call["frames"][0][0, 0], [3, 2, 1])
    np.testing.assert_array_equal(call["frames"][1][0, 0], [9, 8, 7])


def test_fps_falls_back_to_measured_then_30_and_warns(monkeypatch):
    block, model = _make_block()
    warning = MagicMock()
    monkeypatch.setattr(video_classification_module.logger, "warning", warning)

    for frame_number in range(3):
        _run(
            block,
            _make_frame(frame_number, fps=None, measured_fps=None),
            window_size_seconds=0.1,
            sampling_fps=4.0,
        )

    assert model.calls[0]["fps"] == 4.0
    warning.assert_called_once()
    assert "30" in warning.call_args.args[0]

    measured_block, measured_model = _make_block()
    for frame_number in range(5):
        _run(
            measured_block,
            _make_frame(frame_number, fps=None, measured_fps=5.0),
            window_size_seconds=1.0,
            sampling_fps=2.0,
        )
    assert measured_model.calls[0]["fps"] == 2.0
    warning.assert_called_once()


def test_sampling_fps_is_capped_at_source_fps():
    block, model = _make_block()

    _run(block, _make_frame(0, fps=2.0), sampling_fps=10.0)
    _run(block, _make_frame(1, fps=2.0), sampling_fps=10.0)

    assert model.calls[0]["fps"] == 2.0
    assert len(model.calls[0]["frames"]) == 2


def test_maps_dropped_frames_across_a_window_boundary():
    block, model = _make_block(
        responses=[
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "walk"}],
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "run"}],
        ]
    )

    for frame_number in [10, 12, 13, 15, 16, 17]:
        result = _run(block, _make_frame(frame_number))

    assert len(model.calls) == 2
    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 10,
            "end_frame_idx": 12,
            "class_name": "walk",
            "class_id": 0,
        },
        {
            "start_frame_idx": 15,
            "end_frame_idx": 17,
            "class_name": "run",
            "class_id": 1,
        },
    ]


def test_merges_same_class_across_windows_when_gap_is_at_most_stride():
    block, _ = _make_block(
        responses=[
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "walk"}],
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "walk"}],
        ]
    )

    for frame_number in range(8):
        result = _run(block, _make_frame(frame_number))

    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 0,
            "end_frame_idx": 7,
            "class_name": "walk",
            "class_id": 0,
        }
    ]
    assert result["active_classes"] == ["walk"]


def test_preserves_overlapping_classes_and_unions_same_class_overlaps():
    block, _ = _make_block(
        responses=[
            [
                {"start_frame_idx": 0, "end_frame_idx": 1, "class": "walk"},
                {"start_frame_idx": 0, "end_frame_idx": 1, "class": "run"},
                {"start_frame_idx": 1, "end_frame_idx": 1, "class": "walk"},
            ]
        ]
    )

    for frame_number in range(4):
        result = _run(block, _make_frame(frame_number))

    assert [
        (entry.class_name, entry.start_frame_idx) for entry in result["timeline"]
    ] == [
        ("walk", 0),
        ("run", 0),
    ]
    assert result["active_classes"] == ["walk", "run"]


def test_keeps_same_class_ranges_separate_when_gap_exceeds_stride():
    block, _ = _make_block(
        responses=[
            [{"start_frame_idx": 0, "end_frame_idx": 0, "class": "walk"}],
            [{"start_frame_idx": 1, "end_frame_idx": 1, "class": "walk"}],
        ]
    )

    for frame_number in range(8):
        result = _run(block, _make_frame(frame_number))

    assert [
        (entry.start_frame_idx, entry.end_frame_idx) for entry in result["timeline"]
    ] == [
        (0, 0),
        (6, 7),
    ]


def test_open_range_advances_then_closes_at_recorded_endpoint():
    block, _ = _make_block(
        responses=[
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "walk"}],
            [],
        ]
    )

    for frame_number in range(4):
        first_window_result = _run(block, _make_frame(frame_number))
    assert first_window_result["timeline"][0].end_frame_idx == 3
    assert first_window_result["active_classes"] == ["walk"]

    for frame_number in range(4, 7):
        advancing_result = _run(block, _make_frame(frame_number))
    assert advancing_result["timeline"][0].end_frame_idx == 6
    assert first_window_result["timeline"][0].end_frame_idx == 3

    closed_result = _run(block, _make_frame(7))
    assert closed_result["timeline"][0].end_frame_idx == 2
    assert closed_result["active_classes"] == []
    assert _run(block, _make_frame(8))["timeline"][0].end_frame_idx == 2


def test_wire_contract_and_alias_round_trip():
    entity = VideoIntervalClassification(
        start_frame_idx=3,
        end_frame_idx=9,
        class_name="walk",
        class_id=2,
    )

    serialized = serialize_video_multi_label_classification_kind([entity])

    assert set(serialized[0]) == {
        "start_frame_idx",
        "end_frame_idx",
        "class",
        "class_id",
    }
    assert serialized[0]["class"] == "walk"
    assert deserialize_video_multi_label_classification_kind(
        parameter="timeline", value=serialized
    ) == [entity]


@pytest.mark.parametrize("value", [None, {}, ["bad"], [{"class": "walk"}]])
def test_deserializer_rejects_malformed_values(value):
    with pytest.raises(RuntimeInputError):
        deserialize_video_multi_label_classification_kind(
            parameter="timeline", value=value
        )


@pytest.mark.parametrize(
    "reset_kwargs",
    [
        {"class_names": ("run",)},
        {"window_size_seconds": 2.0},
        {"sampling_fps": 1.0},
        {"source_fps": 5.0},
    ],
)
def test_window_defining_input_change_clears_all_state(reset_kwargs):
    block, model = _make_block(
        responses=[[{"start_frame_idx": 0, "end_frame_idx": 1, "class": "walk"}]]
    )
    for frame_number in range(4):
        result = _run(block, _make_frame(frame_number))
    assert result["timeline"]
    assert result["active_classes"] == ["walk"]

    class_names = reset_kwargs.get("class_names", ("walk", "run"))
    window_size_seconds = reset_kwargs.get("window_size_seconds", 1.0)
    sampling_fps = reset_kwargs.get("sampling_fps", 2.0)
    source_fps = reset_kwargs.get("source_fps", 4.0)
    result = _run(
        block,
        _make_frame(4, fps=source_fps),
        class_names=class_names,
        window_size_seconds=window_size_seconds,
        sampling_fps=sampling_fps,
    )

    assert result == {"timeline": [], "active_classes": []}
    assert len(model.calls) == 1


def test_frame_rollback_resets_timeline_and_buffer():
    block, model = _make_block(
        responses=[
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "walk"}],
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "run"}],
        ]
    )
    for frame_number in range(4):
        _run(block, _make_frame(frame_number))

    assert _run(block, _make_frame(0)) == {"timeline": [], "active_classes": []}
    assert len(model.calls) == 1
    for frame_number in range(1, 4):
        result = _run(block, _make_frame(frame_number))

    assert len(model.calls) == 2
    assert [entry.class_name for entry in result["timeline"]] == ["run"]


def test_video_states_are_independent():
    block, model = _make_block(
        responses=[
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "walk"}],
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "run"}],
        ]
    )

    for frame_number in range(3):
        assert _run(block, _make_frame(frame_number, video_id="a"))["timeline"] == []
        assert _run(block, _make_frame(frame_number, video_id="b"))["timeline"] == []
    a_result = _run(block, _make_frame(3, video_id="a"))
    assert [entry.class_name for entry in a_result["timeline"]] == ["walk"]
    assert len(model.calls) == 1
    b_result = _run(block, _make_frame(3, video_id="b"))

    assert [entry.class_name for entry in b_result["timeline"]] == ["run"]
    assert len(model.calls) == 2


def test_video_identifier_can_be_reused_after_rollback_reset():
    block, _ = _make_block(
        responses=[
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "walk"}],
            [{"start_frame_idx": 0, "end_frame_idx": 1, "class": "run"}],
        ]
    )
    for frame_number in range(4):
        _run(block, _make_frame(frame_number, video_id="reused"))
    for frame_number in range(4):
        result = _run(block, _make_frame(frame_number, video_id="reused"))

    assert [entry.class_name for entry in result["timeline"]] == ["run"]


def test_remote_mode_raises():
    block = VideoClassificationModelBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.REMOTE,
    )

    with pytest.raises(NotImplementedError, match="only supports LOCAL"):
        _run(block, _make_frame(0))


def test_tensor_sibling_forwards_tensor_materialised_frames():
    block, model = _make_block(tensor=True)

    _run(
        block,
        _make_frame(0, fps=2.0, tensor_rgb_color=[1, 2, 3]),
        sampling_fps=2.0,
    )
    _run(
        block,
        _make_frame(1, fps=2.0, tensor_rgb_color=[4, 5, 6]),
        sampling_fps=2.0,
    )

    assert all(isinstance(frame, torch.Tensor) for frame in model.calls[0]["frames"])
    assert tuple(model.calls[0]["frames"][0].shape) == (3, 2, 2)
    torch.testing.assert_close(
        model.calls[0]["frames"][0][:, 0, 0],
        torch.tensor([1, 2, 3], dtype=torch.uint8),
    )


def test_tensor_sibling_normalizes_mixed_window_to_rgb_numpy():
    block, model = _make_block(tensor=True)

    _run(
        block,
        _make_frame(0, fps=2.0, tensor_rgb_color=[1, 2, 3]),
        sampling_fps=2.0,
    )
    _run(
        block,
        _make_frame(1, fps=2.0, bgr_color=[6, 5, 4]),
        sampling_fps=2.0,
    )

    frames = model.calls[0]["frames"]
    assert all(isinstance(frame, np.ndarray) for frame in frames)
    assert all(frame.shape == (2, 2, 3) for frame in frames)
    np.testing.assert_array_equal(frames[0][0, 0], [1, 2, 3])
    np.testing.assert_array_equal(frames[1][0, 0], [4, 5, 6])


@pytest.mark.parametrize("tensor_enabled", [False, True])
def test_loader_registers_block_kind_and_codecs_for_both_modes(
    monkeypatch, tensor_enabled
):
    from inference.core.workflows.core_steps import loader

    original = core_env.ENABLE_TENSOR_DATA_REPRESENTATION
    try:
        monkeypatch.setattr(
            core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", tensor_enabled
        )
        reloaded_loader = importlib.reload(loader)

        assert reloaded_loader.VideoClassificationModelBlockV1 in (
            reloaded_loader.load_blocks()
        )
        assert (
            reloaded_loader.VideoClassificationModelBlockV1.__module__.endswith(
                "v1_tensor"
            )
            is tensor_enabled
        )
        assert VIDEO_MULTI_LABEL_CLASSIFICATION_PREDICTION_KIND in (
            reloaded_loader.load_kinds()
        )
        kind_name = VIDEO_MULTI_LABEL_CLASSIFICATION_PREDICTION_KIND.name
        assert (
            reloaded_loader.KINDS_SERIALIZERS[kind_name]
            is serialize_video_multi_label_classification_kind
        )
        assert (
            reloaded_loader.KINDS_DESERIALIZERS[kind_name]
            is deserialize_video_multi_label_classification_kind
        )
    finally:
        monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", original)
        importlib.reload(loader)
