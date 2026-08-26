"""Tests for the stateful Video Segment Classification Model workflow block."""

import importlib
import math
from datetime import datetime
from typing import List, Optional, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import inference.core.env as core_env
from inference_models import VideoSegmentClassificationModel
from inference_models import (
    VideoSegmentClassificationPrediction as ModelVideoSegmentClassificationPrediction,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner
from inference_models.models.cosmos3.cosmos3_video_segment_classification import (
    Cosmos3EdgeVideoSegmentClassification,
)
from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_video_segment_classification_prediction_kind,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.serializers import (
    serialize_video_segment_classification_prediction_kind,
)
from inference.core.workflows.core_steps.models.roboflow.video_segment_classification import (
    v1 as video_classification_module,
)
from inference.core.workflows.core_steps.models.roboflow.video_segment_classification.v1 import (
    BlockManifest,
    VideoSegmentClassificationModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.video_segment_classification.v1_tensor import (
    BlockManifest as TensorBlockManifest,
)
from inference.core.workflows.core_steps.models.roboflow.video_segment_classification.v1_tensor import (
    VideoSegmentClassificationModelBlockV1 as TensorVideoSegmentClassificationModelBlockV1,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoSegmentClassificationPrediction,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    VIDEO_SEGMENT_CLASSIFICATION_PREDICTION_KIND,
)


class _FakeVideoSegmentClassificationModel(VideoSegmentClassificationModel):
    def __init__(
        self,
        responses: Optional[
            List[Union[List[ModelVideoSegmentClassificationPrediction], Exception]]
        ] = None,
    ):
        self.responses = list(responses or [])
        self.calls = []

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        raise NotImplementedError

    def infer(
        self,
        frames,
        class_names=None,
        fps=None,
        **kwargs,
    ):
        recorded_frames = [
            frame.clone() if isinstance(frame, torch.Tensor) else frame.copy()
            for frame in frames
        ]
        self.calls.append(
            {
                "frames": recorded_frames,
                "class_names": list(class_names),
                "fps": fps,
            }
        )
        if not self.responses:
            return []
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _model_segment(
    class_name: str,
    start_frame_idx: int = 0,
    end_frame_idx: int = 0,
) -> ModelVideoSegmentClassificationPrediction:
    return ModelVideoSegmentClassificationPrediction(
        start_frame_idx=start_frame_idx,
        end_frame_idx=end_frame_idx,
        class_name=class_name,
    )


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
    responses: Optional[
        List[Union[List[ModelVideoSegmentClassificationPrediction], Exception]]
    ] = None,
    tensor: bool = False,
):
    block_type = (
        TensorVideoSegmentClassificationModelBlockV1
        if tensor
        else VideoSegmentClassificationModelBlockV1
    )
    block = block_type(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    model = _FakeVideoSegmentClassificationModel(responses=responses)
    block._model = model
    return block, model


def _make_cosmos3_reasoner() -> Cosmos3EdgeReasoner:
    model = MagicMock()
    model.parameters.return_value = iter([torch.tensor(0.0, dtype=torch.bfloat16)])
    return Cosmos3EdgeReasoner(
        model=model,
        processor=MagicMock(),
        device=torch.device("cpu"),
    )


def _run(
    block,
    frame: WorkflowImageData,
    class_names=("walk", "run"),
    window_seconds=1.0,
    stride_seconds=None,
    sample_fps=2.0,
):
    return block.run(
        images=[frame],
        class_names=list(class_names),
        model_id="cosmos-3-edge",
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        sample_fps=sample_fps,
    )[0]


def _timeline_as_dicts(result):
    return [entry.model_dump() for entry in result["timeline"]]


def test_get_model_wraps_hosted_cosmos3_reasoner(monkeypatch):
    from inference_models import AutoModel

    reasoner = _make_cosmos3_reasoner()
    load_model = MagicMock(return_value=reasoner)
    monkeypatch.setattr(AutoModel, "from_pretrained", load_model)
    block = VideoSegmentClassificationModelBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    loaded = block._get_model(model_id="cosmos-3-edge")

    assert isinstance(loaded, Cosmos3EdgeVideoSegmentClassification)
    assert loaded._reasoner is reasoner


def test_get_model_rejects_model_without_video_classification_support(monkeypatch):
    from inference_models import AutoModel

    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        MagicMock(return_value=object()),
    )
    block = VideoSegmentClassificationModelBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    with pytest.raises(
        ValueError,
        match="unrelated-model does not support video classification",
    ):
        block._get_model(model_id="unrelated-model")


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_parses_classes_and_declares_outputs(manifest_type):
    manifest = manifest_type.model_validate(
        {
            "type": "roboflow_core/video_segment_classification_model@v1",
            "name": "video_classifier",
            "images": "$inputs.image",
            "class_names": "walk, run",
        }
    )

    assert manifest.class_names == "walk, run"
    assert manifest.model_id == "cosmos-3-edge"
    assert manifest.window_seconds == 2.0
    assert manifest.stride_seconds is None
    assert manifest.sample_fps == 4.0
    assert manifest_type.get_parameters_accepting_batches() == ["images"]
    assert manifest_type.describe_outputs()[0].kind == [
        VIDEO_SEGMENT_CLASSIFICATION_PREDICTION_KIND
    ]
    assert manifest_type.describe_outputs()[1].kind == [LIST_OF_VALUES_KIND]
    assert manifest_type.describe_outputs()[2].name == "error_status"
    assert manifest_type.describe_outputs()[2].kind == [STRING_KIND]


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_requires_class_names(manifest_type):
    with pytest.raises(Exception):
        manifest_type.model_validate(
            {
                "type": "roboflow_core/video_segment_classification_model@v1",
                "name": "video_classifier",
                "images": "$inputs.image",
            }
        )


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("window_seconds", 0),
        ("window_seconds", -1),
        ("window_seconds", math.inf),
        ("window_seconds", math.nan),
        ("stride_seconds", 0),
        ("stride_seconds", -1),
        ("stride_seconds", math.inf),
        ("stride_seconds", math.nan),
        ("sample_fps", 0),
        ("sample_fps", -1),
        ("sample_fps", math.inf),
        ("sample_fps", math.nan),
    ],
)
def test_manifest_rejects_non_positive_or_non_finite_time_inputs(
    manifest_type, field, value
):
    data = {
        "type": "roboflow_core/video_segment_classification_model@v1",
        "name": "video_classifier",
        "images": "$inputs.image",
        "class_names": ["walk"],
        field: value,
    }

    with pytest.raises(Exception, match="positive and finite"):
        manifest_type.model_validate(data)


def test_first_frame_classifies_a_growing_buffer_and_sends_rgb():
    block, model = _make_block(
        responses=[[_model_segment("walk")]]
    )

    result = _run(block, _make_frame(0, bgr_color=[1, 2, 3]))

    assert [entry.class_name for entry in result["timeline"]] == ["walk"]
    assert result["active_classes"] == ["walk"]
    assert result["error_status"] == ""
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["class_names"] == ["walk", "run"]
    assert call["fps"] == 2.0
    assert len(call["frames"]) == 1
    np.testing.assert_array_equal(call["frames"][0][0, 0], [3, 2, 1])


def test_default_stride_fires_every_half_window():
    block, model = _make_block()

    call_counts = []
    for frame_number in range(6):
        _run(block, _make_frame(frame_number))
        call_counts.append(len(model.calls))

    assert call_counts == [1, 1, 2, 2, 3, 3]


def test_window_stride_produces_non_overlapping_call_cadence():
    block, model = _make_block()

    fired_at = []
    for frame_number in range(9):
        previous_call_count = len(model.calls)
        _run(
            block,
            _make_frame(frame_number),
            stride_seconds=1.0,
        )
        if len(model.calls) > previous_call_count:
            fired_at.append(frame_number)

    assert fired_at == [0, 4, 8]


def test_default_stride_retains_overlapping_sampled_frames():
    block, model = _make_block()

    for frame_number in range(5):
        _run(block, _make_frame(frame_number))

    second_call_values = [frame[0, 0, 2] for frame in model.calls[1]["frames"]]
    third_call_values = [frame[0, 0, 2] for frame in model.calls[2]["frames"]]
    assert second_call_values == [0, 2]
    assert third_call_values == [2, 4]
    np.testing.assert_array_equal(
        model.calls[1]["frames"][-1], model.calls[2]["frames"][0]
    )


def test_fps_falls_back_to_measured_then_30_and_warns(monkeypatch):
    block, model = _make_block()
    warning = MagicMock()
    monkeypatch.setattr(video_classification_module.logger, "warning", warning)

    for frame_number in range(3):
        _run(
            block,
            _make_frame(frame_number, fps=None, measured_fps=None),
            window_seconds=0.1,
            sample_fps=4.0,
        )

    assert model.calls[0]["fps"] == 4.0
    warning.assert_called_once()
    assert "30" in warning.call_args.args[0]

    measured_block, measured_model = _make_block()
    for frame_number in range(5):
        _run(
            measured_block,
            _make_frame(frame_number, fps=None, measured_fps=5.0),
            window_seconds=1.0,
            sample_fps=2.0,
        )
    assert measured_model.calls[0]["fps"] == 2.0
    warning.assert_called_once()


def test_sample_fps_is_capped_at_source_fps():
    block, model = _make_block()

    _run(block, _make_frame(0, fps=2.0), sample_fps=10.0)
    _run(block, _make_frame(1, fps=2.0), sample_fps=10.0)

    assert model.calls[0]["fps"] == 2.0
    assert model.calls[1]["fps"] == 2.0
    assert len(model.calls[1]["frames"]) == 2


def test_dropped_frame_gap_fires_once_and_maps_the_current_buffer():
    block, model = _make_block(
        responses=[
            [_model_segment("walk")],
            [],
            [_model_segment("run")],
        ]
    )

    _run(block, _make_frame(10))
    _run(block, _make_frame(12))
    calls_before_gap = len(model.calls)
    result = _run(block, _make_frame(100))

    assert len(model.calls) == calls_before_gap + 1
    assert len(model.calls[-1]["frames"]) == 1
    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 10,
            "end_frame_idx": 10,
            "class_name": "walk",
            "class_id": 0,
        },
        {
            "start_frame_idx": 100,
            "end_frame_idx": 100,
            "class_name": "run",
            "class_id": 1,
        },
    ]


def test_merges_same_class_across_windows_when_gap_is_at_most_stride():
    block, _ = _make_block(
        responses=[
            [_model_segment("walk")],
            [_model_segment("walk", end_frame_idx=1)],
        ]
    )

    for frame_number in range(4):
        result = _run(block, _make_frame(frame_number))

    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 0,
            "end_frame_idx": 3,
            "class_name": "walk",
            "class_id": 0,
        }
    ]
    assert result["active_classes"] == ["walk"]


def test_preserves_overlapping_classes_and_unions_same_class_overlaps():
    block, _ = _make_block(
        responses=[
            [
                _model_segment("walk"),
                _model_segment("run"),
                _model_segment("walk"),
            ]
        ]
    )

    result = _run(block, _make_frame(0))

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
            [_model_segment("walk")],
            [],
            [_model_segment("walk", start_frame_idx=1, end_frame_idx=1)],
        ]
    )

    for frame_number in range(6):
        result = _run(block, _make_frame(frame_number))

    assert [
        (entry.start_frame_idx, entry.end_frame_idx) for entry in result["timeline"]
    ] == [
        (0, 0),
        (4, 5),
    ]


def test_open_range_advances_then_closes_at_recorded_endpoint():
    block, _ = _make_block(
        responses=[
            [_model_segment("walk")],
            [],
        ]
    )

    first_window_result = _run(block, _make_frame(0))
    assert first_window_result["timeline"][0].end_frame_idx == 0
    assert first_window_result["active_classes"] == ["walk"]

    advancing_result = _run(block, _make_frame(1))
    assert advancing_result["timeline"][0].end_frame_idx == 1
    assert first_window_result["timeline"][0].end_frame_idx == 0

    closed_result = _run(block, _make_frame(2))
    assert closed_result["timeline"][0].end_frame_idx == 0
    assert closed_result["active_classes"] == []
    assert _run(block, _make_frame(3))["timeline"][0].end_frame_idx == 0


def test_wire_contract_and_alias_round_trip():
    entity = VideoSegmentClassificationPrediction(
        start_frame_idx=3,
        end_frame_idx=9,
        class_name="walk",
        class_id=2,
    )

    serialized = serialize_video_segment_classification_prediction_kind([entity])

    assert set(serialized[0]) == {
        "start_frame_idx",
        "end_frame_idx",
        "class",
        "class_id",
    }
    assert serialized[0]["class"] == "walk"
    assert deserialize_video_segment_classification_prediction_kind(
        parameter="timeline", value=serialized
    ) == [entity]


@pytest.mark.parametrize("value", [None, {}, ["bad"], [{"class": "walk"}]])
def test_deserializer_rejects_malformed_values(value):
    with pytest.raises(RuntimeInputError):
        deserialize_video_segment_classification_prediction_kind(
            parameter="timeline", value=value
        )


@pytest.mark.parametrize(
    "reset_kwargs",
    [
        {"class_names": ("run",)},
        {"window_seconds": 2.0},
        {"stride_seconds": 0.5},
        {"sample_fps": 1.0},
        {"source_fps": 5.0},
    ],
)
def test_window_defining_input_change_clears_all_state(reset_kwargs):
    block, model = _make_block(responses=[[_model_segment("walk")]])
    result = _run(block, _make_frame(0))
    assert result["timeline"]
    assert result["active_classes"] == ["walk"]

    class_names = reset_kwargs.get("class_names", ("walk", "run"))
    window_seconds = reset_kwargs.get("window_seconds", 1.0)
    stride_seconds = reset_kwargs.get("stride_seconds")
    sample_fps = reset_kwargs.get("sample_fps", 2.0)
    source_fps = reset_kwargs.get("source_fps", 4.0)
    result = _run(
        block,
        _make_frame(1, fps=source_fps),
        class_names=class_names,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
        sample_fps=sample_fps,
    )

    assert result == {"timeline": [], "active_classes": [], "error_status": ""}
    assert len(model.calls) == 2


def test_model_failure_preserves_state_and_later_success_resumes(monkeypatch):
    block, model = _make_block(
        responses=[
            [_model_segment("walk")],
            RuntimeError("temporary model failure"),
            [_model_segment("run", start_frame_idx=1, end_frame_idx=1)],
        ]
    )
    warning = MagicMock()
    monkeypatch.setattr(video_classification_module.logger, "warning", warning)

    initial = _run(block, _make_frame(0))
    _run(block, _make_frame(1))
    failed = _run(block, _make_frame(2))

    assert initial["active_classes"] == ["walk"]
    assert failed["error_status"] == "temporary model failure"
    assert [entry.class_name for entry in failed["timeline"]] == ["walk"]
    assert failed["active_classes"] == ["walk"]
    assert block._video_bookkeeping["stream-0"].timeline[0].end_frame_idx == 0
    warning.assert_called_once()
    assert warning.call_args.kwargs["exc_info"] is True

    after_failure = _run(block, _make_frame(3))
    assert after_failure["error_status"] == ""
    assert after_failure["active_classes"] == ["walk"]
    assert len(model.calls) == 2

    resumed = _run(block, _make_frame(4))

    assert resumed["error_status"] == ""
    assert [entry.class_name for entry in resumed["timeline"]] == ["walk", "run"]
    assert resumed["active_classes"] == ["run"]
    assert len(model.calls) == 3


def test_error_status_is_empty_on_frames_without_a_failed_call():
    block, model = _make_block()

    first = _run(block, _make_frame(0))
    ordinary = _run(block, _make_frame(1))

    assert first["error_status"] == ""
    assert ordinary["error_status"] == ""
    assert len(model.calls) == 1


def test_frame_rollback_resets_timeline_and_buffer():
    block, model = _make_block(
        responses=[
            [_model_segment("walk")],
            [_model_segment("run")],
        ]
    )
    _run(block, _make_frame(0))
    _run(block, _make_frame(1))

    result = _run(block, _make_frame(0))

    assert len(model.calls) == 2
    assert [entry.class_name for entry in result["timeline"]] == ["run"]


def test_video_states_are_independent():
    block, model = _make_block(
        responses=[
            [_model_segment("walk")],
            [_model_segment("run")],
        ]
    )

    a_result = _run(block, _make_frame(0, video_id="a"))
    assert [entry.class_name for entry in a_result["timeline"]] == ["walk"]
    assert len(model.calls) == 1
    b_result = _run(block, _make_frame(0, video_id="b"))

    assert [entry.class_name for entry in b_result["timeline"]] == ["run"]
    assert len(model.calls) == 2


def test_video_identifier_can_be_reused_after_rollback_reset():
    block, _ = _make_block(
        responses=[
            [_model_segment("walk")],
            [_model_segment("run")],
        ]
    )
    _run(block, _make_frame(0, video_id="reused"))
    _run(block, _make_frame(1, video_id="reused"))
    result = _run(block, _make_frame(0, video_id="reused"))

    assert [entry.class_name for entry in result["timeline"]] == ["run"]


def test_remote_mode_raises():
    block = VideoSegmentClassificationModelBlockV1(
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
        sample_fps=2.0,
    )
    _run(
        block,
        _make_frame(1, fps=2.0, tensor_rgb_color=[4, 5, 6]),
        sample_fps=2.0,
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
        sample_fps=2.0,
    )
    _run(
        block,
        _make_frame(1, fps=2.0, bgr_color=[6, 5, 4]),
        sample_fps=2.0,
    )

    frames = model.calls[1]["frames"]
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

        assert reloaded_loader.VideoSegmentClassificationModelBlockV1 in (
            reloaded_loader.load_blocks()
        )
        assert (
            reloaded_loader.VideoSegmentClassificationModelBlockV1.__module__.endswith(
                "v1_tensor"
            )
            is tensor_enabled
        )
        assert VIDEO_SEGMENT_CLASSIFICATION_PREDICTION_KIND in (
            reloaded_loader.load_kinds()
        )
        kind_name = VIDEO_SEGMENT_CLASSIFICATION_PREDICTION_KIND.name
        assert (
            reloaded_loader.KINDS_SERIALIZERS[kind_name]
            is serialize_video_segment_classification_prediction_kind
        )
        assert (
            reloaded_loader.KINDS_DESERIALIZERS[kind_name]
            is deserialize_video_segment_classification_prediction_kind
        )
    finally:
        monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", original)
        importlib.reload(loader)
