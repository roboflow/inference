"""Tests for the stateful Action Recognition Model workflow block."""

import importlib
import math
from datetime import datetime
from typing import List, Optional, Union
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

import inference.core.env as core_env
from inference.core.workflows.core_steps.common.deserializers import (
    deserialize_action_recognition_prediction_kind,
)
from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.common.serializers import (
    serialize_action_recognition_prediction_kind,
)
from inference.core.workflows.core_steps.models.roboflow.action_recognition import (
    v1 as video_classification_module,
)
from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1 import (
    ActionRecognitionModelBlockV1,
    BlockManifest,
)
from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1_tensor import (
    ActionRecognitionModelBlockV1 as TensorActionRecognitionModelBlockV1,
)
from inference.core.workflows.core_steps.models.roboflow.action_recognition.v1_tensor import (
    BlockManifest as TensorBlockManifest,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.entities.base import (
    ActionRecognitionPrediction,
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    ACTION_RECOGNITION_PREDICTION_KIND,
    STRING_KIND,
)
from inference_models import ActionRecognitionModel
from inference_models import (
    ActionRecognitionPrediction as ModelActionRecognitionPrediction,
)
from inference_models import VideoSampling
from inference_models.models.base.action_recognition import WHOLE_VIDEO_MODE
from inference_models.models.cosmos3.cosmos3_action_recognition import (
    Cosmos3EdgeActionRecognition,
)
from inference_models.models.cosmos3.cosmos3_reasoner_hf import Cosmos3EdgeReasoner


class _FakeActionRecognitionModel(ActionRecognitionModel):
    # A plain class attribute overrides the ABC property so tests can set
    # per-instance temporal contracts.
    video_sampling = VideoSampling()

    def __init__(
        self,
        responses: Optional[
            List[Union[List[ModelActionRecognitionPrediction], Exception]]
        ] = None,
        class_names: Optional[List[str]] = None,
    ):
        self.responses = list(responses or [])
        self.calls = []
        self._class_names = class_names

    @property
    def class_names(self) -> Optional[List[str]]:
        return self._class_names

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
                "class_names": (list(class_names) if class_names is not None else None),
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
) -> ModelActionRecognitionPrediction:
    return ModelActionRecognitionPrediction(
        start_frame_idx=start_frame_idx,
        end_frame_idx=end_frame_idx,
        class_name=class_name,
    )


def _make_frame(
    frame_number: int,
    video_id: str = "stream-0",
    fps: Optional[float] = 4.0,
    measured_fps: Optional[float] = None,
    frame_timestamp: datetime = datetime(2024, 1, 1),
    bgr_color: Optional[List[int]] = None,
    tensor_rgb_color: Optional[List[int]] = None,
) -> WorkflowImageData:
    metadata = VideoMetadata(
        video_identifier=video_id,
        frame_number=frame_number,
        fps=fps,
        measured_fps=measured_fps,
        frame_timestamp=frame_timestamp,
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
        workflow_root_ancestor_metadata=ImageParentMetadata(
            parent_id=f"{video_id}:root"
        ),
        video_metadata=metadata,
        **image_kwargs,
    )


def _make_block(
    responses: Optional[
        List[Union[List[ModelActionRecognitionPrediction], Exception]]
    ] = None,
    tensor: bool = False,
    model_class_names: Optional[List[str]] = None,
):
    block_type = (
        TensorActionRecognitionModelBlockV1 if tensor else ActionRecognitionModelBlockV1
    )
    block = block_type(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )
    model = _FakeActionRecognitionModel(
        responses=responses,
        class_names=model_class_names,
    )
    block._model = model
    block._current_model_id = "cosmos-3-edge"
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
    class_filter=("walk", "run"),
    window_seconds=1.0,
    stride_seconds=0.5,
    sample_fps=2.0,
    min_frames=1,
):
    # The temporal contract travels with the model.
    if block._model is not None:
        block._model.video_sampling = VideoSampling(
            window_seconds=window_seconds,
            sample_fps=sample_fps,
            min_frames=min_frames,
        )
    return block.run(
        images=[frame],
        class_filter=list(class_filter) if class_filter is not None else None,
        model_id="cosmos-3-edge",
        stride_seconds=stride_seconds,
    )[0]


def _timeline_as_dicts(result):
    return [entry.model_dump() for entry in result["timeline"]]


def test_get_model_wraps_hosted_cosmos3_reasoner(monkeypatch):
    from inference_models import AutoModel

    reasoner = _make_cosmos3_reasoner()
    load_model = MagicMock(return_value=reasoner)
    monkeypatch.setattr(AutoModel, "from_pretrained", load_model)
    block = ActionRecognitionModelBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    loaded = block._get_model(model_id="cosmos-3-edge")

    assert isinstance(loaded, Cosmos3EdgeActionRecognition)
    assert loaded._reasoner is reasoner


def test_get_model_rejects_model_without_video_classification_support(monkeypatch):
    from inference_models import AutoModel

    monkeypatch.setattr(
        AutoModel,
        "from_pretrained",
        MagicMock(return_value=object()),
    )
    block = ActionRecognitionModelBlockV1(
        model_manager=MagicMock(),
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    with pytest.raises(
        ValueError,
        match="unrelated-model does not support action recognition",
    ):
        block._get_model(model_id="unrelated-model")


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_parses_class_filter_and_declares_outputs(manifest_type):
    manifest = manifest_type.model_validate(
        {
            "type": "roboflow_core/roboflow_action_recognition_model@v1",
            "name": "video_classifier",
            "images": "$inputs.image",
            "class_filter": ["walk", "run"],
            "model_id": "cosmos-3-edge",
        }
    )

    assert manifest.class_filter == ["walk", "run"]
    assert manifest.model_id == "cosmos-3-edge"
    assert manifest.stride_seconds is None
    assert manifest_type.get_parameters_accepting_batches() == ["images"]
    outputs = manifest_type.describe_outputs()
    assert [output.name for output in outputs] == ["timeline", "error_status"]
    assert outputs[0].kind == [ACTION_RECOGNITION_PREDICTION_KIND]
    assert outputs[1].kind == [STRING_KIND]


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_allows_omitted_class_filter(manifest_type):
    manifest = manifest_type.model_validate(
        {
            "type": "roboflow_core/roboflow_action_recognition_model@v1",
            "name": "video_classifier",
            "images": "$inputs.image",
            "model_id": "cosmos-3-edge",
        }
    )

    assert manifest.class_filter is None


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_requires_model_id(manifest_type):
    with pytest.raises(Exception):
        manifest_type.model_validate(
            {
                "type": "roboflow_core/roboflow_action_recognition_model@v1",
                "name": "video_classifier",
                "images": "$inputs.image",
                "class_filter": ["walk"],
            }
        )


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stride_seconds", 0),
        ("stride_seconds", -1),
        ("stride_seconds", math.inf),
        ("stride_seconds", math.nan),
    ],
)
def test_manifest_rejects_non_positive_or_non_finite_time_inputs(
    manifest_type, field, value
):
    data = {
        "type": "roboflow_core/roboflow_action_recognition_model@v1",
        "name": "video_classifier",
        "images": "$inputs.image",
        "model_id": "cosmos-3-edge",
        "class_filter": ["walk"],
        field: value,
    }

    with pytest.raises(Exception, match="positive and finite"):
        manifest_type.model_validate(data)


def test_first_frame_anchors_cadence_and_first_fire_waits_for_stride():
    block, model = _make_block(responses=[[_model_segment("walk")]])

    first_result = _run(block, _make_frame(10))

    assert first_result["timeline"] == []
    assert model.calls == []

    _run(block, _make_frame(11))
    assert model.calls == []

    first_fire_result = _run(block, _make_frame(12))

    assert len(model.calls) == 1
    assert [entry.class_name for entry in first_fire_result["timeline"]] == ["walk"]


def test_first_scheduled_classification_sends_rgb_stride_buffer():
    block, model = _make_block(responses=[[_model_segment("walk")]])

    _run(block, _make_frame(0, bgr_color=[1, 2, 3]))
    result = _run(block, _make_frame(2))

    assert [entry.class_name for entry in result["timeline"]] == ["walk"]
    assert result["error_status"] == ""
    assert len(model.calls) == 1
    call = model.calls[0]
    assert call["class_names"] == ["walk", "run"]
    assert call["fps"] == 2.0
    assert len(call["frames"]) == 2
    np.testing.assert_array_equal(call["frames"][0][0, 0], [3, 2, 1])


def test_open_vocabulary_keeps_arbitrary_labels_with_negative_class_ids():
    block, model = _make_block(
        responses=[
            [
                _model_segment("opening a door"),
                _model_segment("sitting down"),
            ]
        ],
        model_class_names=None,
    )

    _run(block, _make_frame(0), class_filter=None)
    result = _run(block, _make_frame(2), class_filter=None)

    assert model.calls[0]["class_names"] is None
    assert [entry.class_name for entry in result["timeline"]] == [
        "opening a door",
        "sitting down",
    ]
    assert [entry.class_id for entry in result["timeline"]] == [-1, -1]


def test_model_vocabulary_sets_class_ids_when_class_filter_is_omitted():
    block, model = _make_block(
        responses=[[_model_segment("run")]],
        model_class_names=["walk", "run"],
    )

    _run(block, _make_frame(0), class_filter=None)
    result = _run(block, _make_frame(2), class_filter=None)

    assert model.calls[0]["class_names"] is None
    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 0,
            "end_frame_idx": 0,
            "class_name": "run",
            "class_id": 1,
        }
    ]


def test_class_filter_reaches_the_model_but_supplies_no_class_ids():
    # The filter still forms the prompt vocabulary. It is not a class list,
    # though: a model carrying none has no ids to report, and borrowing the
    # filter's order would invent one.
    block, model = _make_block(
        responses=[[_model_segment("run")]],
        model_class_names=None,
    )

    _run(block, _make_frame(0), class_filter=("run", "walk"))
    result = _run(block, _make_frame(2), class_filter=("run", "walk"))

    assert model.calls[0]["class_names"] == ["run", "walk"]
    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 0,
            "end_frame_idx": 0,
            "class_name": "run",
            "class_id": -1,
        }
    ]


def test_baked_vocabulary_keeps_stable_ids_when_class_filter_is_set():
    block, model = _make_block(
        responses=[[_model_segment("a"), _model_segment("b")]],
        model_class_names=["a", "b", "c"],
    )

    _run(block, _make_frame(0), class_filter=("b",))
    result = _run(block, _make_frame(2), class_filter=("b",))

    assert model.calls[0]["class_names"] == ["b"]
    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 0,
            "end_frame_idx": 0,
            "class_name": "b",
            "class_id": 1,
        }
    ]


def test_half_window_stride_fires_twice_per_window():
    block, model = _make_block()

    call_counts = []
    for frame_number in range(7):
        _run(block, _make_frame(frame_number))
        call_counts.append(len(model.calls))

    assert call_counts == [0, 0, 1, 1, 2, 2, 3]


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

    assert fired_at == [4, 8]


def test_default_stride_retains_overlapping_sampled_frames():
    block, model = _make_block()

    for frame_number in range(5):
        _run(block, _make_frame(frame_number))

    first_call_values = [frame[0, 0, 2] for frame in model.calls[0]["frames"]]
    second_call_values = [frame[0, 0, 2] for frame in model.calls[1]["frames"]]
    assert first_call_values == [0, 2]
    assert second_call_values == [2, 4]
    np.testing.assert_array_equal(
        model.calls[0]["frames"][-1], model.calls[1]["frames"][0]
    )


def test_missing_fps_pins_30_fallback_and_warns_once(monkeypatch):
    block, _ = _make_block()
    warning = MagicMock()
    monkeypatch.setattr(video_classification_module.logger, "warning", warning)

    for frame_number in range(3):
        _run(block, _make_frame(frame_number, fps=None, measured_fps=None))

    assert block._video_bookkeeping["stream-0"].source_fps == 30.0
    warning.assert_called_once()
    assert "30" in warning.call_args.args[0]


@pytest.mark.parametrize(
    ("fps", "measured_fps", "expected_fps"),
    [(25.0, None, 25.0), (25.0, 24.5, 25.0)],
)
def test_valid_metadata_fps_pins_immediately_without_estimation(
    monkeypatch, fps, measured_fps, expected_fps
):
    block, _ = _make_block()
    info = MagicMock()
    warning = MagicMock()
    monkeypatch.setattr(video_classification_module.logger, "info", info)
    monkeypatch.setattr(video_classification_module.logger, "warning", warning)

    _run(
        block,
        _make_frame(0, fps=fps, measured_fps=measured_fps),
    )

    bookkeeping = block._video_bookkeeping["stream-0"]
    assert bookkeeping.source_fps == expected_fps
    info.assert_not_called()
    warning.assert_not_called()


@pytest.mark.parametrize(
    ("reset_frame_number", "reset_window_seconds"),
    [(0, 1.0), (9, 2.0)],
)
def test_reset_re_resolves_source_fps(reset_frame_number, reset_window_seconds):
    block, _ = _make_block()
    for frame_number in range(9):
        _run(block, _make_frame(frame_number, fps=25.0))
    previous_bookkeeping = block._video_bookkeeping["stream-0"]
    assert previous_bookkeeping.source_fps == 25.0

    _run(
        block,
        _make_frame(reset_frame_number, fps=None, measured_fps=None),
        window_seconds=reset_window_seconds,
    )

    reset_bookkeeping = block._video_bookkeeping["stream-0"]
    assert reset_bookkeeping is not previous_bookkeeping
    assert reset_bookkeeping.source_fps == 30.0


def test_run_default_stride_equals_the_window():
    block, model = _make_block()
    model.video_sampling = VideoSampling(
        window_seconds=1.0, sample_fps=4.0, min_frames=1
    )
    for frame_number in range(9):
        block.run(
            images=[_make_frame(frame_number, fps=4.0)],
            class_filter=["walk", "run"],
            model_id="cosmos-3-edge",
        )

    assert len(model.calls) == 2
    bookkeeping = block._video_bookkeeping["stream-0"]
    assert bookkeeping.last_fire_frame_number == 8


def test_min_frames_gates_the_first_fire():
    block, model = _make_block()
    model.video_sampling = VideoSampling(
        window_seconds=2.0, sample_fps=4.0, min_frames=4
    )

    fired_at = []
    for frame_number in range(8):
        previous_call_count = len(model.calls)
        block.run(
            images=[_make_frame(frame_number, fps=4.0)],
            class_filter=["walk", "run"],
            model_id="cosmos-3-edge",
            stride_seconds=0.25,
        )
        if len(model.calls) > previous_call_count:
            fired_at.append(frame_number)

    # Stride alone allows a fire every frame; the model's 4-frame floor
    # delays the first fire until the buffer holds 4 samples.
    assert fired_at[0] == 3
    assert len(model.calls[0]["frames"]) == 4


def test_tracked_video_count_is_bounded():
    block, model = _make_block()

    # A crop step mints a fresh video identifier per detection per frame.
    for frame_number in range(video_classification_module.MAX_TRACKED_VIDEOS + 20):
        _run(block, _make_frame(0, video_id=f"stream-{frame_number}"))

    assert (
        len(block._video_bookkeeping) <= video_classification_module.MAX_TRACKED_VIDEOS
    )


def test_fractional_sampling_stride_keeps_the_true_sample_rate():
    # 30 fps at sample_fps 4 gives a 7.5-frame stride. The float grid
    # alternates 7- and 8-frame spacing; anchoring at sampled integers
    # would round every step to 8 and skew the model's clock.
    block, _ = _make_block()
    for frame_number in range(31):
        _run(
            block,
            _make_frame(frame_number, fps=30.0),
            window_seconds=10.0,
            sample_fps=4.0,
        )

    sampled_numbers = [
        number for number, _ in block._video_bookkeeping["stream-0"].sampled
    ]
    assert sampled_numbers == [0, 8, 15, 23, 30]


def test_sample_fps_is_capped_at_source_fps():
    block, model = _make_block()

    _run(block, _make_frame(0, fps=2.0), sample_fps=10.0)
    _run(block, _make_frame(1, fps=2.0), sample_fps=10.0)
    _run(block, _make_frame(2, fps=2.0), sample_fps=10.0)

    assert model.calls[0]["fps"] == 2.0
    assert model.calls[1]["fps"] == 2.0
    assert len(model.calls[1]["frames"]) == 2


def test_dropped_frame_gap_fires_once_and_maps_the_current_buffer():
    block, model = _make_block(
        responses=[
            [_model_segment("walk")],
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
            "class_id": -1,
        },
        {
            "start_frame_idx": 100,
            "end_frame_idx": 100,
            "class_name": "run",
            "class_id": -1,
        },
    ]


def test_merges_same_class_across_windows_when_gap_is_at_most_stride():
    block, _ = _make_block(
        responses=[
            [_model_segment("walk")],
            [_model_segment("walk", end_frame_idx=1)],
        ]
    )

    for frame_number in range(5):
        result = _run(block, _make_frame(frame_number))

    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 0,
            "end_frame_idx": 4,
            "class_name": "walk",
            "class_id": -1,
        }
    ]


def test_merges_reports_separated_by_a_fractional_stride_boundary():
    # Observed live: with sampling stride 7.5, samples land alternately 7 and
    # 8 frames apart, so adjacent windows' reports sit ceil(stride) apart and
    # a float tolerance misses the merge by under one frame.
    block, model = _make_block(
        responses=[
            [_model_segment("walk")],
            [_model_segment("walk", start_frame_idx=0, end_frame_idx=6)],
        ]
    )

    for frame_number in range(61):
        result = _run(
            block,
            _make_frame(frame_number, fps=30.0),
            window_seconds=2.0,
            stride_seconds=1.0,
            sample_fps=4.0,
        )

    assert [call["fps"] for call in model.calls] == [4.0, 4.0]
    # The first fire covers absolute frame 0; the second report starts at
    # frame 8 — a gap of exactly ceil(7.5) that must merge into ONE entry.
    # The second report ends at absolute frame 53. The output does not extend
    # that model-confirmed endpoint to the current frame.
    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 0,
            "end_frame_idx": 53,
            "class_name": "walk",
            "class_id": -1,
        }
    ]


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

    _run(block, _make_frame(0))
    result = _run(block, _make_frame(2))

    # Two classes covering the same range have no order between them, so the
    # assertion is about which ranges survive, not how they are sequenced.
    assert sorted(
        (entry.class_name, entry.start_frame_idx) for entry in result["timeline"]
    ) == [("run", 0), ("walk", 0)]


def test_keeps_same_class_ranges_separate_when_gap_exceeds_stride():
    block, _ = _make_block(
        responses=[
            [_model_segment("walk")],
            [],
            [_model_segment("walk", start_frame_idx=1, end_frame_idx=1)],
        ]
    )

    for frame_number in range(7):
        result = _run(block, _make_frame(frame_number))

    assert [
        (entry.start_frame_idx, entry.end_frame_idx) for entry in result["timeline"]
    ] == [
        (0, 0),
        (6, 6),
    ]


def test_timeline_advances_only_as_model_confirmed_monotone_staircase():
    block, _ = _make_block(
        responses=[
            [_model_segment("walk")],
            [_model_segment("walk")],
            [_model_segment("walk")],
        ]
    )

    output_endpoints = []
    stored_endpoints = []
    for frame_number in range(8):
        result = _run(block, _make_frame(frame_number))
        if result["timeline"]:
            output_endpoints.append(result["timeline"][0].end_frame_idx)
            stored_endpoints.append(
                block._video_bookkeeping["stream-0"].timeline[0].end_frame_idx
            )

    assert output_endpoints == [0, 0, 2, 2, 4, 4]
    assert stored_endpoints == output_endpoints


def test_wire_contract_and_alias_round_trip():
    entity = ActionRecognitionPrediction(
        start_frame_idx=3,
        end_frame_idx=9,
        class_name="walk",
        class_id=2,
    )

    serialized = serialize_action_recognition_prediction_kind([entity])

    assert set(serialized[0]) == {
        "start_frame_idx",
        "end_frame_idx",
        "class",
        "class_id",
    }
    assert serialized[0]["class"] == "walk"
    assert deserialize_action_recognition_prediction_kind(
        parameter="timeline", value=serialized
    ) == [entity]


@pytest.mark.parametrize("value", [None, {}, ["bad"], [{"class": "walk"}]])
def test_deserializer_rejects_malformed_values(value):
    with pytest.raises(RuntimeInputError):
        deserialize_action_recognition_prediction_kind(
            parameter="timeline", value=value
        )


@pytest.mark.parametrize(
    "reset_kwargs",
    [
        {"class_filter": ("run",)},
        {"window_seconds": 2.0},
        {"stride_seconds": 1.0},
    ],
)
def test_window_defining_input_change_clears_all_state(reset_kwargs):
    block, model = _make_block(responses=[[_model_segment("walk")]])
    _run(block, _make_frame(0))
    _run(block, _make_frame(1))
    result = _run(block, _make_frame(2))
    assert result["timeline"]

    class_filter = reset_kwargs.get("class_filter", ("walk", "run"))
    window_seconds = reset_kwargs.get("window_seconds", 1.0)
    stride_seconds = reset_kwargs.get("stride_seconds", 0.5)
    reset_result = _run(
        block,
        _make_frame(3),
        class_filter=class_filter,
        window_seconds=window_seconds,
        stride_seconds=stride_seconds,
    )

    assert reset_result["timeline"] == []
    assert reset_result["error_status"] == ""
    assert len(model.calls) == 1

    reset_fire_frame = 3 + max(1, round(stride_seconds * 4.0))
    for frame_number in range(4, reset_fire_frame + 1):
        result = _run(
            block,
            _make_frame(frame_number),
            class_filter=class_filter,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
        )

    assert result["timeline"] == []
    assert len(model.calls) == 2


def test_source_fps_jitter_does_not_reset_state():
    # measured_fps jitters per frame on live streams and is never consumed;
    # its arrival mid-stream must neither re-pin the fps nor clear state.
    block, model = _make_block(responses=[[_model_segment("walk")]])
    _run(block, _make_frame(0, fps=None, measured_fps=None))

    for frame_number, measured in ((1, 24.7), (2, 25.3), (3, 24.9)):
        _run(
            block,
            _make_frame(frame_number, fps=None, measured_fps=measured),
        )
    result = _run(
        block,
        _make_frame(15, fps=None, measured_fps=24.9),
    )

    bookkeeping = block._video_bookkeeping["stream-0"]
    assert [entry.class_name for entry in result["timeline"]] == ["walk"]
    assert bookkeeping.source_fps == 30.0
    assert len(model.calls) == 1


def test_processing_paced_measured_fps_is_never_consumed():
    # WebRTC ACK pacing makes measured_fps track model latency (~0.05 fps
    # observed live); consuming it collapses the stride to one frame.
    block, _ = _make_block()
    for frame_number in range(3):
        _run(block, _make_frame(frame_number, fps=None, measured_fps=0.05))

    assert block._video_bookkeeping["stream-0"].source_fps == 30.0


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

    _run(block, _make_frame(0))
    _run(block, _make_frame(1))
    _run(block, _make_frame(2))
    _run(block, _make_frame(3))
    failed = _run(block, _make_frame(4))

    assert failed["error_status"] == "temporary model failure"
    assert [entry.class_name for entry in failed["timeline"]] == ["walk"]
    assert block._video_bookkeeping["stream-0"].timeline[0].end_frame_idx == 0
    warning.assert_called_once()
    assert warning.call_args.kwargs["exc_info"] is True

    after_failure = _run(block, _make_frame(5))
    assert after_failure["error_status"] == ""
    assert len(model.calls) == 2

    resumed = _run(block, _make_frame(6))

    assert resumed["error_status"] == ""
    assert [entry.class_name for entry in resumed["timeline"]] == ["walk", "run"]
    assert len(model.calls) == 3


def test_error_status_is_empty_on_frames_without_a_failed_call():
    block, model = _make_block()

    anchor = _run(block, _make_frame(0))
    _run(block, _make_frame(1))
    first = _run(block, _make_frame(2))
    ordinary = _run(block, _make_frame(3))

    assert anchor["error_status"] == ""
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
    _run(block, _make_frame(2))

    _run(block, _make_frame(0))
    _run(block, _make_frame(1))
    result = _run(block, _make_frame(2))

    assert len(model.calls) == 2
    assert [entry.class_name for entry in result["timeline"]] == ["run"]


def test_video_states_are_independent():
    block, model = _make_block(
        responses=[
            [_model_segment("walk")],
            [_model_segment("run")],
        ]
    )

    _run(block, _make_frame(0, video_id="a"))
    a_result = _run(block, _make_frame(2, video_id="a"))
    assert [entry.class_name for entry in a_result["timeline"]] == ["walk"]
    assert len(model.calls) == 1
    _run(block, _make_frame(0, video_id="b"))
    b_result = _run(block, _make_frame(2, video_id="b"))

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
    _run(block, _make_frame(2, video_id="reused"))
    _run(block, _make_frame(0, video_id="reused"))
    _run(block, _make_frame(1, video_id="reused"))
    result = _run(block, _make_frame(2, video_id="reused"))

    assert [entry.class_name for entry in result["timeline"]] == ["run"]


def test_remote_mode_raises():
    block = ActionRecognitionModelBlockV1(
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
        model.calls[0]["frames"][0][:, 0, 0].cpu(),
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

        assert reloaded_loader.ActionRecognitionModelBlockV1 in (
            reloaded_loader.load_blocks()
        )
        assert (
            reloaded_loader.ActionRecognitionModelBlockV1.__module__.endswith(
                "v1_tensor"
            )
            is tensor_enabled
        )
        assert ACTION_RECOGNITION_PREDICTION_KIND in (reloaded_loader.load_kinds())
        kind_name = ACTION_RECOGNITION_PREDICTION_KIND.name
        assert (
            reloaded_loader.KINDS_SERIALIZERS[kind_name]
            is serialize_action_recognition_prediction_kind
        )
        assert (
            reloaded_loader.KINDS_DESERIALIZERS[kind_name]
            is deserialize_action_recognition_prediction_kind
        )
    finally:
        monkeypatch.setattr(core_env, "ENABLE_TENSOR_DATA_REPRESENTATION", original)
        importlib.reload(loader)


def test_the_block_refuses_a_whole_video_model() -> None:
    # Whole-video training spans a clip, and a stream never ends, so the
    # block declines rather than sample the model a way it never saw.
    block, model = _make_block(responses=[[_model_segment("walk", 0, 3)]])
    model.video_sampling = VideoSampling(mode=WHOLE_VIDEO_MODE)
    frame = _make_frame(frame_number=1)

    with pytest.raises(ValueError, match="whole videos"):
        block.run(
            images=[frame],
            class_filter=None,
            model_id="cosmos-3-edge",
            stride_seconds=None,
        )

    # It refuses before it asks the model for anything.
    assert model.calls == []


def test_a_trained_model_keeps_its_rate_when_the_stream_is_slower() -> None:
    # Training drew 4 timestamps a second and repeated the frame under each
    # one. Capping to the source would hand the model a shorter input,
    # stamped at a rate it never trained on.
    block, model = _make_block(responses=[[_model_segment("walk")]])
    model.video_sampling = VideoSampling(
        window_seconds=2.0, sample_fps=4.0, min_frames=1, max_frames=8
    )

    for frame_number in range(5):
        block.run(
            images=[_make_frame(frame_number, fps=2.0)],
            class_filter=None,
            model_id="cosmos-3-edge",
            stride_seconds=2.0,
        )

    assert model.calls, "the block never fired"
    # 2 fps source against a 4 fps contract: two samples per arriving frame.
    assert model.calls[-1]["fps"] == 4.0
    assert len(model.calls[-1]["frames"]) == 8


def test_an_untrained_model_still_caps_at_the_source_rate() -> None:
    # Zero-shot has no training to reproduce, so a repeat buys nothing.
    block, model = _make_block(responses=[[_model_segment("walk")]])
    model.video_sampling = VideoSampling(
        window_seconds=2.0, sample_fps=4.0, min_frames=1
    )

    for frame_number in range(5):
        block.run(
            images=[_make_frame(frame_number, fps=2.0)],
            class_filter=None,
            model_id="cosmos-3-edge",
            stride_seconds=2.0,
        )

    assert model.calls[-1]["fps"] == 2.0
    assert len(model.calls[-1]["frames"]) == 4
