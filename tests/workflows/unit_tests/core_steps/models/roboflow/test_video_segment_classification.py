"""Tests for the stateful Video Segment Classification Model workflow block."""

import importlib
import math
from datetime import datetime
from typing import List, Optional, Union
from unittest.mock import MagicMock
from uuid import UUID

import numpy as np
import pytest
import torch

import inference.core.env as core_env
from inference_models import VideoSampling, VideoSegmentClassificationModel
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
from inference.core.workflows.execution_engine.constants import (
    CLASS_NAMES_KEY,
    CLASSIFICATION_STYLE_KEY,
    CLASSIFICATION_STYLE_MODEL,
    IMAGE_DIMENSIONS_KEY,
    INFERENCE_ID_KEY,
    PARENT_ID_KEY,
    PREDICTION_TYPE_KEY,
    ROOT_PARENT_ID_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    Batch,
    ImageParentMetadata,
    VideoSegmentClassificationPrediction,
    VideoMetadata,
    WorkflowImageData,
)
from inference.core.workflows.execution_engine.entities.types import (
    CLASSIFICATION_PREDICTION_KIND,
    STRING_KIND,
    VIDEO_SEGMENT_CLASSIFICATION_PREDICTION_KIND,
)
from inference_models.models.base.classification import (
    MultiLabelClassificationPrediction,
)


class _FakeVideoSegmentClassificationModel(VideoSegmentClassificationModel):
    # A plain class attribute overrides the ABC property so tests can set
    # per-instance temporal contracts.
    video_sampling = VideoSampling()

    def __init__(
        self,
        responses: Optional[
            List[Union[List[ModelVideoSegmentClassificationPrediction], Exception]]
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
                "class_names": (
                    list(class_names) if class_names is not None else None
                ),
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
        video_metadata=metadata,
        **image_kwargs,
    )


def _make_block(
    responses: Optional[
        List[Union[List[ModelVideoSegmentClassificationPrediction], Exception]]
    ] = None,
    tensor: bool = False,
    model_class_names: Optional[List[str]] = None,
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
    model = _FakeVideoSegmentClassificationModel(
        responses=responses,
        class_names=model_class_names,
    )
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


def _window_class_names(result):
    return result["window_classes"]["predicted_classes"]


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
        match="unrelated-model does not support video segment classification",
    ):
        block._get_model(model_id="unrelated-model")


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_parses_class_filter_and_declares_outputs(manifest_type):
    manifest = manifest_type.model_validate(
        {
            "type": "roboflow_core/video_segment_classification_model@v1",
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
    assert manifest_type.describe_outputs()[0].kind == [
        VIDEO_SEGMENT_CLASSIFICATION_PREDICTION_KIND
    ]
    assert manifest_type.describe_outputs()[1].kind == [
        CLASSIFICATION_PREDICTION_KIND
    ]
    assert manifest_type.describe_outputs()[1].name == "window_classes"
    assert manifest_type.describe_outputs()[2].name == "error_status"
    assert manifest_type.describe_outputs()[2].kind == [STRING_KIND]


@pytest.mark.parametrize("manifest_type", [BlockManifest, TensorBlockManifest])
def test_manifest_allows_omitted_class_filter(manifest_type):
    manifest = manifest_type.model_validate(
        {
            "type": "roboflow_core/video_segment_classification_model@v1",
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
                "type": "roboflow_core/video_segment_classification_model@v1",
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
        "type": "roboflow_core/video_segment_classification_model@v1",
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
    assert _window_class_names(first_result) == []
    assert first_result["window_classes"]["predictions"] == {}
    assert model.calls == []

    _run(block, _make_frame(11))
    assert model.calls == []

    first_fire_result = _run(block, _make_frame(12))

    assert len(model.calls) == 1
    assert [entry.class_name for entry in first_fire_result["timeline"]] == [
        "walk"
    ]


def test_first_scheduled_classification_sends_rgb_stride_buffer():
    block, model = _make_block(
        responses=[[_model_segment("walk")]]
    )

    _run(block, _make_frame(0, bgr_color=[1, 2, 3]))
    result = _run(block, _make_frame(2))

    assert [entry.class_name for entry in result["timeline"]] == ["walk"]
    assert _window_class_names(result) == ["walk"]
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
    assert _window_class_names(result) == ["opening a door", "sitting down"]
    assert result["window_classes"]["predictions"] == {
        "opening a door": {"confidence": 1.0, "class_id": -1},
        "sitting down": {"confidence": 1.0, "class_id": -1},
    }


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
    assert _window_class_names(result) == ["run"]


def test_class_filter_is_prompt_vocabulary_when_model_vocabulary_is_omitted():
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
            "class_id": 0,
        }
    ]
    assert result["window_classes"]["predictions"]["run"]["class_id"] == 0


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
    assert result["window_classes"]["predictions"] == {
        "b": {"confidence": 1.0, "class_id": 1}
    }


def test_numpy_window_classes_uses_legacy_multi_label_classification_shape():
    block, _ = _make_block(
        responses=[[_model_segment("walk")]],
        tensor=False,
    )
    _run(block, _make_frame(0))
    frame = _make_frame(2)

    result = _run(block, frame)

    window_classes = result["window_classes"]
    assert window_classes["image"] == {"width": 2, "height": 2}
    assert window_classes["predictions"] == {
        "walk": {"confidence": 1.0, "class_id": 0}
    }
    assert window_classes["predicted_classes"] == ["walk"]
    assert window_classes["prediction_type"] == "classification"
    assert window_classes["parent_id"] == "stream-0:2"
    assert window_classes["root_parent_id"] == "stream-0:2"
    assert UUID(window_classes["inference_id"]).version == 4


def test_tensor_window_classes_uses_dense_vocabulary():
    block, _ = _make_block(
        responses=[[_model_segment("run"), _model_segment("walk")]],
        tensor=True,
        model_class_names=["walk", "run", "idle"],
    )
    _run(
        block,
        _make_frame(0, tensor_rgb_color=[1, 2, 3]),
        class_filter=None,
    )
    frame = _make_frame(2, tensor_rgb_color=[4, 5, 6])

    result = _run(block, frame, class_filter=None)

    window_classes = result["window_classes"]
    assert isinstance(window_classes, MultiLabelClassificationPrediction)
    assert window_classes.class_ids.dtype is torch.long
    assert window_classes.confidence.dtype is torch.float32
    torch.testing.assert_close(
        window_classes.class_ids.cpu(),
        torch.tensor([0, 1], dtype=torch.long),
    )
    torch.testing.assert_close(
        window_classes.confidence.cpu(),
        torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32),
    )
    metadata = window_classes.image_metadata
    assert metadata[CLASS_NAMES_KEY] == {0: "walk", 1: "run", 2: "idle"}
    assert metadata[CLASSIFICATION_STYLE_KEY] == CLASSIFICATION_STYLE_MODEL
    assert metadata[PREDICTION_TYPE_KEY] == "classification"
    assert metadata[IMAGE_DIMENSIONS_KEY] == [2, 2]
    assert metadata[PARENT_ID_KEY] == "stream-0:2"
    assert metadata[ROOT_PARENT_ID_KEY] == "stream-0:2"
    assert UUID(metadata[INFERENCE_ID_KEY]).version == 4


def test_tensor_window_classes_uses_open_vocabulary_label_order():
    block, _ = _make_block(
        responses=[
            [
                _model_segment("opening a door"),
                _model_segment("sitting down"),
            ]
        ],
        tensor=True,
    )
    _run(
        block,
        _make_frame(0, tensor_rgb_color=[1, 2, 3]),
        class_filter=None,
    )

    result = _run(
        block,
        _make_frame(2, tensor_rgb_color=[4, 5, 6]),
        class_filter=None,
    )

    window_classes = result["window_classes"]
    assert window_classes.image_metadata[CLASS_NAMES_KEY] == {
        0: "opening a door",
        1: "sitting down",
    }
    torch.testing.assert_close(
        window_classes.class_ids.cpu(),
        torch.tensor([0, 1], dtype=torch.long),
    )
    torch.testing.assert_close(
        window_classes.confidence.cpu(),
        torch.ones(2, dtype=torch.float32),
    )


def test_tensor_window_classes_empty_fire_uses_empty_tensors():
    from inference.core.workflows.core_steps.visualizations.classification_label.v1_tensor import (
        to_legacy_classification_prediction,
    )

    block, model = _make_block(
        responses=[[]],
        tensor=True,
        model_class_names=["walk", "run"],
    )
    _run(
        block,
        _make_frame(0, tensor_rgb_color=[1, 2, 3]),
        class_filter=None,
    )

    result = _run(
        block,
        _make_frame(2, tensor_rgb_color=[4, 5, 6]),
        class_filter=None,
    )

    assert len(model.calls) == 1
    window_classes = result["window_classes"]
    assert window_classes.class_ids.dtype is torch.long
    assert window_classes.class_ids.numel() == 0
    assert window_classes.confidence.dtype is torch.float32
    assert window_classes.confidence.numel() == 0
    assert window_classes.image_metadata[CLASS_NAMES_KEY] == {
        0: "walk",
        1: "run",
    }
    assert window_classes.image_metadata[IMAGE_DIMENSIONS_KEY] == [2, 2]
    assert to_legacy_classification_prediction(window_classes) == {
        "image": {"width": 2, "height": 2},
        "predictions": {},
        "predicted_classes": [],
    }


def test_tensor_window_classes_round_trips_to_legacy_multi_label_shape():
    from inference.core.workflows.core_steps.visualizations.classification_label.v1_tensor import (
        to_legacy_classification_prediction,
    )

    block, _ = _make_block(
        responses=[[_model_segment("run"), _model_segment("walk")]],
        tensor=True,
        model_class_names=["walk", "run", "idle"],
    )
    _run(
        block,
        _make_frame(0, tensor_rgb_color=[1, 2, 3]),
        class_filter=None,
    )
    result = _run(
        block,
        _make_frame(2, tensor_rgb_color=[4, 5, 6]),
        class_filter=None,
    )

    legacy = to_legacy_classification_prediction(result["window_classes"])

    assert legacy == {
        "image": {"width": 2, "height": 2},
        "predictions": {
            "walk": {"confidence": 1.0, "class_id": 0},
            "run": {"confidence": 1.0, "class_id": 1},
            "idle": {"confidence": 0.0, "class_id": 2},
        },
        "predicted_classes": ["walk", "run"],
    }


def test_window_classes_keep_alive_through_negative_fire_and_errors_then_clear(
    monkeypatch,
):
    block, model = _make_block(
        responses=[
            [_model_segment("walk"), _model_segment("walk", 1, 1)],
            RuntimeError("temporary model failure"),
            [],
        ]
    )
    monkeypatch.setattr(video_classification_module.logger, "warning", MagicMock())

    assert _window_class_names(_run(block, _make_frame(0))) == []
    assert _window_class_names(_run(block, _make_frame(1))) == []

    fired = _run(block, _make_frame(2))
    assert _window_class_names(fired) == ["walk"]
    assert fired["timeline"][0].end_frame_idx == 2

    held = _run(block, _make_frame(3))
    assert _window_class_names(held) == ["walk"]

    errored = _run(block, _make_frame(4))
    assert errored["error_status"]
    assert _window_class_names(errored) == ["walk"]

    _run(block, _make_frame(5))
    negative_fire = _run(block, _make_frame(6))
    assert negative_fire["error_status"] == ""
    assert _window_class_names(negative_fire) == ["walk"]

    _run(block, _make_frame(7))
    # window_frames=4 and ceil(sampling_stride)=2, so frame 8 is the final
    # mergeable frame for the range that ends at frame 2.
    last_mergeable_fire = _run(block, _make_frame(8))
    assert _window_class_names(last_mergeable_fire) == ["walk"]

    cleared = _run(block, _make_frame(9))
    assert _window_class_names(cleared) == []
    assert len(model.calls) == 4


def test_default_stride_fires_every_half_window():
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


def test_flush_classifies_short_clip_below_default_window():
    block, model = _make_block(
        responses=[[_model_segment("walk", start_frame_idx=0, end_frame_idx=3)]]
    )

    for frame_number in range(4):
        result = _run(
            block,
            _make_frame(frame_number),
            window_seconds=2.0,
            stride_seconds=None,
            sample_fps=4.0,
            min_frames=4,
        )

    assert block.is_stream_pipelined() is True
    assert block.stream_pipeline_depth() == 0
    assert result["timeline"] == []
    assert model.calls == []

    flushed = block.flush_stream_pipeline_outputs()

    assert len(flushed) == 1
    indices, outputs = flushed[0]
    assert indices == [(0,)]
    assert len(outputs) == 1
    assert len(model.calls) == 1
    assert len(model.calls[0]["frames"]) == 4
    assert [frame[0, 0, 2] for frame in model.calls[0]["frames"]] == [
        0,
        1,
        2,
        3,
    ]
    assert _timeline_as_dicts(outputs[0]) == [
        {
            "start_frame_idx": 0,
            "end_frame_idx": 3,
            "class_name": "walk",
            "class_id": 0,
        }
    ]
    assert _window_class_names(outputs[0]) == ["walk"]


def test_flush_classifies_tail_after_a_scheduled_fire():
    block, model = _make_block(
        responses=[
            [_model_segment("walk", start_frame_idx=0, end_frame_idx=2)],
            [_model_segment("walk", start_frame_idx=0, end_frame_idx=3)],
        ]
    )

    for frame_number in range(4):
        result = _run(
            block,
            _make_frame(frame_number),
            sample_fps=4.0,
        )

    assert len(model.calls) == 1
    assert result["timeline"][0].end_frame_idx == 2

    flushed = block.flush_stream_pipeline_outputs()

    assert len(model.calls) == 2
    assert len(flushed) == 1
    assert flushed[0][0] == [(0,)]
    assert flushed[0][1][0]["timeline"][0].end_frame_idx == 3


def test_flush_immediately_after_fire_without_new_sample_returns_empty():
    block, model = _make_block(responses=[[_model_segment("walk")]])

    for frame_number in range(3):
        _run(
            block,
            _make_frame(frame_number),
            sample_fps=4.0,
        )

    assert len(model.calls) == 1
    assert block.flush_stream_pipeline_outputs() == []
    assert len(model.calls) == 1


def test_flush_skips_buffer_below_model_minimum():
    block, model = _make_block()

    for frame_number in range(3):
        _run(
            block,
            _make_frame(frame_number),
            window_seconds=2.0,
            stride_seconds=None,
            sample_fps=4.0,
            min_frames=4,
        )

    assert block.flush_stream_pipeline_outputs() == []
    assert model.calls == []


def test_flush_emits_only_video_with_a_pending_tail():
    block, model = _make_block(
        responses=[
            [_model_segment("walk", start_frame_idx=0, end_frame_idx=2)],
            [_model_segment("run", start_frame_idx=0, end_frame_idx=1)],
        ]
    )
    model.video_sampling = VideoSampling(
        window_seconds=1.0,
        sample_fps=4.0,
        min_frames=1,
    )
    first_batch = Batch.init(
        content=[
            _make_frame(0, video_id="a"),
            _make_frame(0, video_id="b"),
        ],
        indices=[(0,), (1,)],
    )
    second_batch = Batch.init(
        content=[
            _make_frame(2, video_id="a"),
            _make_frame(1, video_id="b"),
        ],
        indices=[(0,), (1,)],
    )

    block.run(
        images=first_batch,
        class_filter=["walk", "run"],
        model_id="cosmos-3-edge",
        stride_seconds=0.5,
    )
    block.run(
        images=second_batch,
        class_filter=["walk", "run"],
        model_id="cosmos-3-edge",
        stride_seconds=0.5,
    )
    assert len(model.calls) == 1

    flushed = block.flush_stream_pipeline_outputs()

    assert len(model.calls) == 2
    assert len(flushed) == 1
    assert flushed[0][0] == [(1,)]
    assert [entry.class_name for entry in flushed[0][1][0]["timeline"]] == [
        "run"
    ]


def test_consecutive_flush_calls_do_not_repeat_the_tail():
    block, model = _make_block(
        responses=[[_model_segment("walk", start_frame_idx=0, end_frame_idx=1)]]
    )
    for frame_number in range(2):
        _run(
            block,
            _make_frame(frame_number),
            window_seconds=2.0,
            stride_seconds=None,
            sample_fps=4.0,
            min_frames=2,
        )

    first_flush = block.flush_stream_pipeline_outputs()
    second_flush = block.flush_stream_pipeline_outputs()

    assert len(first_flush) == 1
    assert second_flush == []
    assert len(model.calls) == 1
    block.close_stream_pipeline()
    assert block._video_bookkeeping == {}


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

    for frame_number in range(5):
        result = _run(block, _make_frame(frame_number))

    assert _timeline_as_dicts(result) == [
        {
            "start_frame_idx": 0,
            "end_frame_idx": 4,
            "class_name": "walk",
            "class_id": 0,
        }
    ]
    assert _window_class_names(result) == ["walk"]


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
            "class_id": 0,
        }
    ]
    assert _window_class_names(result) == ["walk"]


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

    assert [
        (entry.class_name, entry.start_frame_idx) for entry in result["timeline"]
    ] == [
        ("walk", 0),
        ("run", 0),
    ]
    assert _window_class_names(result) == ["walk", "run"]


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
    assert _window_class_names(result) == ["walk"]

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
    assert _window_class_names(reset_result) == []
    assert reset_result["window_classes"]["predictions"] == {}
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
    initial = _run(block, _make_frame(2))
    _run(block, _make_frame(3))
    failed = _run(block, _make_frame(4))

    assert _window_class_names(initial) == ["walk"]
    assert failed["error_status"] == "temporary model failure"
    assert [entry.class_name for entry in failed["timeline"]] == ["walk"]
    assert _window_class_names(failed) == ["walk"]
    assert block._video_bookkeeping["stream-0"].timeline[0].end_frame_idx == 0
    warning.assert_called_once()
    assert warning.call_args.kwargs["exc_info"] is True

    after_failure = _run(block, _make_frame(5))
    assert after_failure["error_status"] == ""
    assert _window_class_names(after_failure) == ["walk"]
    assert len(model.calls) == 2

    resumed = _run(block, _make_frame(6))

    assert resumed["error_status"] == ""
    assert [entry.class_name for entry in resumed["timeline"]] == ["walk", "run"]
    assert _window_class_names(resumed) == ["walk", "run"]
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
