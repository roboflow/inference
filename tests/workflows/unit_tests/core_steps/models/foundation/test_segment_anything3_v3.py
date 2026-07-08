"""Unit tests for SAM3 v3 block class_mapping feature."""

import threading
from concurrent.futures import Future
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import supervision as sv

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.segment_anything3 import (
    v3 as segment_anything3_v3,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3.v3 import (
    BlockManifest,
    SegmentAnything3BlockV3,
)
from inference.core.workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)


def _make_detections(class_names: list[str]) -> sv.Detections:
    n = len(class_names)
    return sv.Detections(
        xyxy=np.array([[0, 0, 10, 10]] * n, dtype=np.float32),
        confidence=np.array([0.9] * n, dtype=np.float32),
        data={"class_name": np.array(class_names)},
    )


def _make_result(class_names: list[str]) -> list[dict]:
    return [{"predictions": _make_detections(class_names)}]


@pytest.fixture
def mock_workflow_image_data():
    img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    return WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="test"),
        numpy_image=img,
    )


# --- Manifest tests ---


def test_manifest_parsing_with_class_mapping():
    """Test that BlockManifest accepts the class_mapping field."""
    data = {
        "type": "roboflow_core/sam3@v3",
        "name": "my_sam3_step",
        "images": "$inputs.image",
        "class_names": ["cat", "dog"],
        "class_mapping": {"cat": "gato", "dog": "perro"},
    }
    result = BlockManifest.model_validate(data)
    assert result.class_mapping == {"cat": "gato", "dog": "perro"}


def test_manifest_parsing_without_class_mapping():
    """Test that class_mapping is optional and defaults to None."""
    data = {
        "type": "roboflow_core/sam3@v3",
        "name": "my_sam3_step",
        "images": "$inputs.image",
        "class_names": ["cat", "dog"],
    }
    result = BlockManifest.model_validate(data)
    assert result.class_mapping is None


# --- _apply_class_mapping unit tests ---


def test_apply_class_mapping_full():
    """Test remapping all class names."""
    result = _make_result(["cat", "dog"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(
        result, {"cat": "gato", "dog": "perro"}
    )
    assert list(mapped[0]["predictions"].data["class_name"]) == ["gato", "perro"]


def test_apply_class_mapping_partial():
    """Test remapping only some class names, leaving others unchanged."""
    result = _make_result(["cat", "dog", "bird"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {"cat": "gato"})
    assert list(mapped[0]["predictions"].data["class_name"]) == [
        "gato",
        "dog",
        "bird",
    ]


def test_apply_class_mapping_no_matching_keys():
    """Test that unmatched mapping keys leave predictions unchanged."""
    result = _make_result(["cat", "dog"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {"fish": "pez"})
    assert list(mapped[0]["predictions"].data["class_name"]) == ["cat", "dog"]


def test_apply_class_mapping_multiple_images():
    """Test remapping across multiple images in a batch."""
    result = [
        {"predictions": _make_detections(["cat"])},
        {"predictions": _make_detections(["dog"])},
    ]
    mapped = SegmentAnything3BlockV3._apply_class_mapping(
        result, {"cat": "gato", "dog": "perro"}
    )
    assert list(mapped[0]["predictions"].data["class_name"]) == ["gato"]
    assert list(mapped[1]["predictions"].data["class_name"]) == ["perro"]


def test_apply_class_mapping_empty_result():
    """Test that an empty result list is handled gracefully."""
    result = []
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {"cat": "gato"})
    assert mapped == []


def test_apply_class_mapping_empty_mapping():
    """Test that an empty mapping leaves predictions unchanged."""
    result = _make_result(["cat", "dog"])
    mapped = SegmentAnything3BlockV3._apply_class_mapping(result, {})
    assert list(mapped[0]["predictions"].data["class_name"]) == ["cat", "dog"]


# --- Block-level run() tests ---


@patch.object(SegmentAnything3BlockV3, "run_locally")
def test_run_with_class_mapping_remaps_predictions(
    mock_run_locally, mock_workflow_image_data
):
    """Test that block.run() applies class_mapping to predictions from run_locally."""
    mock_run_locally.return_value = _make_result(["cat", "dog"])
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_id="sam3/sam3_final",
        class_names=["cat", "dog"],
        confidence=0.5,
        class_mapping={"cat": "gato", "dog": "perro"},
    )

    assert list(result[0]["predictions"].data["class_name"]) == ["gato", "perro"]


@patch.object(SegmentAnything3BlockV3, "run_locally")
def test_run_without_class_mapping_leaves_predictions_unchanged(
    mock_run_locally, mock_workflow_image_data
):
    """Test that block.run() without class_mapping does not alter predictions."""
    mock_run_locally.return_value = _make_result(["cat", "dog"])
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_id="sam3/sam3_final",
        class_names=["cat", "dog"],
        confidence=0.5,
    )

    assert list(result[0]["predictions"].data["class_name"]) == ["cat", "dog"]


@patch.object(SegmentAnything3BlockV3, "run_locally")
def test_run_with_partial_class_mapping(mock_run_locally, mock_workflow_image_data):
    """Test that block.run() with partial class_mapping only remaps matched classes."""
    mock_run_locally.return_value = _make_result(["cat", "dog", "bird"])
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.LOCAL,
    )

    result = block.run(
        images=[mock_workflow_image_data],
        model_id="sam3/sam3_final",
        class_names=["cat", "dog", "bird"],
        confidence=0.5,
        class_mapping={"cat": "gato"},
    )

    assert list(result[0]["predictions"].data["class_name"]) == [
        "gato",
        "dog",
        "bird",
    ]


# --- Remote stream pipelining tests ---


class _FakeStreamImage:
    def __init__(self, tag: str) -> None:
        self.tag = tag


def _make_remote_block() -> SegmentAnything3BlockV3:
    return SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=StepExecutionMode.REMOTE,
    )


def _stub_execute_remote_inference(
    block: SegmentAnything3BlockV3,
    calls: Optional[List[dict]] = None,
) -> None:
    def _stub(**kwargs):
        if calls is not None:
            calls.append({**kwargs, "thread_id": threading.get_ident()})
        return [{"predictions": f"seg-{kwargs['images'][0].tag}"}]

    block._execute_remote_inference = _stub


def _run_remotely(block: SegmentAnything3BlockV3, images: list):
    return block.run_remotely(
        images=images,
        model_id="sam3/sam3_final",
        class_names=["cat"],
        confidence=0.5,
        per_class_confidence=None,
        apply_nms=True,
        nms_iou_threshold=0.9,
        output_format="rle",
        class_mapping=None,
    )


@pytest.mark.parametrize(
    "sam3_exec_mode, execution_mode, depth, expected_pipelined, expected_depth",
    [
        ("local", StepExecutionMode.REMOTE, 4, True, 3),
        ("remote", StepExecutionMode.REMOTE, 4, False, 0),
        ("local", StepExecutionMode.LOCAL, 4, False, 0),
        ("local", StepExecutionMode.REMOTE, 1, False, 0),
        ("remote", StepExecutionMode.LOCAL, 1, False, 0),
    ],
)
def test_stream_pipeline_protocol_gating(
    monkeypatch,
    sam3_exec_mode: str,
    execution_mode: StepExecutionMode,
    depth: int,
    expected_pipelined: bool,
    expected_depth: int,
):
    """SAM3_EXEC_MODE=remote must disable pipelining even with REMOTE + depth>1."""
    monkeypatch.setattr(segment_anything3_v3, "SAM3_EXEC_MODE", sam3_exec_mode)
    monkeypatch.setattr(
        segment_anything3_v3, "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH", depth
    )
    block = SegmentAnything3BlockV3(
        model_manager=MagicMock(),
        api_key="test_key",
        step_execution_mode=execution_mode,
    )

    assert block.is_stream_pipelined() is expected_pipelined
    assert block.can_activate_stream_pipeline() is expected_pipelined
    assert block.stream_pipeline_depth() == expected_depth
    assert block.defers_downstream_execution() is True


def test_pipelined_run_remotely_returns_prediction_futures(monkeypatch):
    """Pipelined run_remotely returns per-image futures and queues a pending request."""
    monkeypatch.setattr(segment_anything3_v3, "SAM3_EXEC_MODE", "local")
    monkeypatch.setattr(
        segment_anything3_v3, "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH", 4
    )
    block = _make_remote_block()
    _stub_execute_remote_inference(block)

    result = _run_remotely(block, images=[_FakeStreamImage("a")])

    assert len(result) == 1
    assert isinstance(result[0]["predictions"], Future)
    assert result[0]["predictions"].result(timeout=5) == "seg-a"
    assert block._remote_pipeline.pending_requests == 1

    flushed = block.flush_stream_pipeline_outputs()
    assert flushed == [([(0,)], [{"predictions": "seg-a"}])]
    assert block.flush_stream_pipeline_outputs() == []

    block.close_stream_pipeline()
    assert block._remote_pipeline is None


def test_pipelined_run_remotely_flushes_frames_in_fifo_order(monkeypatch):
    """Two pipelined frames flush one per call, oldest first."""
    monkeypatch.setattr(segment_anything3_v3, "SAM3_EXEC_MODE", "local")
    monkeypatch.setattr(
        segment_anything3_v3, "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH", 4
    )
    block = _make_remote_block()
    _stub_execute_remote_inference(block)

    first = _run_remotely(block, images=[_FakeStreamImage("a")])
    second = _run_remotely(block, images=[_FakeStreamImage("b")])

    assert first[0]["predictions"].result(timeout=5) == "seg-a"
    assert second[0]["predictions"].result(timeout=5) == "seg-b"
    assert block._remote_pipeline.pending_requests == 2

    first_flush = block.flush_stream_pipeline_outputs()
    second_flush = block.flush_stream_pipeline_outputs()

    assert first_flush == [([(0,)], [{"predictions": "seg-a"}])]
    assert second_flush == [([(0,)], [{"predictions": "seg-b"}])]
    assert block.flush_stream_pipeline_outputs() == []

    block.close_stream_pipeline()
    assert block._remote_pipeline is None


def test_non_pipelined_run_remotely_executes_synchronously(monkeypatch):
    """With depth 1, run_remotely runs inline on the caller thread."""
    monkeypatch.setattr(segment_anything3_v3, "SAM3_EXEC_MODE", "local")
    monkeypatch.setattr(
        segment_anything3_v3, "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH", 1
    )
    block = _make_remote_block()
    calls = []
    _stub_execute_remote_inference(block, calls=calls)

    result = _run_remotely(block, images=[_FakeStreamImage("a")])

    assert result == [{"predictions": "seg-a"}]
    assert len(calls) == 1
    assert calls[0]["thread_id"] == threading.get_ident()
    assert block._remote_pipeline is None


def test_run_defers_class_mapping_into_pipelined_remote_task(monkeypatch):
    """With pipelining active, run() must not remap futures itself — the mapping
    travels into the worker task applied by _execute_remote_inference."""
    monkeypatch.setattr(segment_anything3_v3, "SAM3_EXEC_MODE", "local")
    monkeypatch.setattr(
        segment_anything3_v3, "WORKFLOWS_REMOTE_EXECUTION_PIPELINE_DEPTH", 4
    )
    block = _make_remote_block()
    calls = []
    _stub_execute_remote_inference(block, calls=calls)

    result = block.run(
        images=[_FakeStreamImage("a")],
        model_id="sam3/sam3_final",
        class_names=["cat"],
        confidence=0.5,
        class_mapping={"cat": "gato"},
    )

    assert len(result) == 1
    assert isinstance(result[0]["predictions"], Future)
    assert result[0]["predictions"].result(timeout=5) == "seg-a"
    assert len(calls) == 1
    assert calls[0]["class_mapping"] == {"cat": "gato"}

    block.close_stream_pipeline()
    assert block._remote_pipeline is None
