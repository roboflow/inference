"""Versioned window snapshots and visualization connections."""

from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.serializers_tensor import (
    serialise_native_classification,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v2 import (
    ActionRecognitionModelBlockV2,
)
from roboflow_workflows.core_steps.models.roboflow.action_recognition.v2_tensor import (
    ActionRecognitionModelBlockV2 as TensorActionRecognitionModelBlockV2,
)
from roboflow_workflows.core_steps.visualizations.classification_label import (
    v1 as classification_label_v1,
)
from roboflow_workflows.core_steps.visualizations.classification_label import (
    v1_tensor as classification_label_v1_tensor,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)

from inference_models import VideoSampling
from tests.workflows.unit_tests.core_steps.models.roboflow.action_recognition.test_v1 import (
    _FakeActionRecognitionModel,
)
from tests.workflows.unit_tests.core_steps.models.roboflow.action_recognition.test_v1 import (
    _make_block as _make_v1_block,
)
from tests.workflows.unit_tests.core_steps.models.roboflow.action_recognition.test_v1 import (
    _make_frame,
    _model_segment,
    _run,
)


def _make_block(responses=None, tensor=False, model_class_names=None):
    block_type = (
        TensorActionRecognitionModelBlockV2 if tensor else ActionRecognitionModelBlockV2
    )
    block = block_type(api_key=None, step_execution_mode=StepExecutionMode.LOCAL)
    model = _FakeActionRecognitionModel(
        responses=responses, class_names=model_class_names
    )
    block._model = model
    block._current_model_id = "cosmos-3-edge"
    return block, model


def _predicted_classes(result):
    prediction = result["latest_predictions"]
    if not isinstance(prediction, dict):
        prediction = serialise_native_classification(prediction)
    return prediction["predicted_classes"]


@pytest.mark.parametrize("tensor", [False, True])
def test_latest_predictions_hold_until_the_next_call_and_clear_on_error(tensor):
    block, _ = _make_block(
        responses=[
            [
                _model_segment("walk", 0, 0),
                _model_segment("run", 1, 1),
                _model_segment("walk", 1, 1),
                _model_segment("jump", 1, 1),
            ],
            RuntimeError("model unavailable"),
        ],
        tensor=tensor,
    )
    color = {"tensor_rgb_color": [1, 2, 3]} if tensor else {}

    results = [_run(block, _make_frame(n, **color)) for n in range(6)]

    # Calls fire on frames 2 and 4; "jump" is outside the class filter.
    assert [_predicted_classes(r) for r in results] == [
        [],
        [],
        ["run", "walk"],
        ["run", "walk"],
        [],
        [],
    ]
    assert results[4]["error_status"] == "model unavailable"


@pytest.mark.parametrize(
    "tensor, visualizer_module",
    [(False, classification_label_v1), (True, classification_label_v1_tensor)],
)
@pytest.mark.parametrize("has_actions", [True, False])
def test_latest_predictions_render_with_classification_label_visualization(
    tensor, visualizer_module, has_actions
):
    block, _ = _make_block(
        responses=[[_model_segment("walk", 1, 1)] if has_actions else []],
        tensor=tensor,
    )
    color = {"tensor_rgb_color": [0, 0, 0]} if tensor else {"bgr_color": [0, 0, 0]}
    for n in range(3):
        result = _run(block, _make_frame(n, **color))
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="canvas"),
        numpy_image=np.zeros((120, 200, 3), dtype=np.uint8),
    )
    manifest = visualizer_module.ClassificationLabelManifest(
        type="roboflow_core/classification_label_visualization@v1",
        name="labels",
        image="$inputs.image",
        predictions="$steps.actions.latest_predictions",
        text="Class",
    )
    configuration = manifest.model_dump(
        exclude={"type", "name", "image", "predictions"}
    )

    output = visualizer_module.ClassificationLabelVisualizationBlockV1().run(
        image=image, predictions=result["latest_predictions"], **configuration
    )

    assert bool(np.any(output["image"].numpy_image)) is has_actions


@pytest.mark.parametrize("tensor", [False, True])
def test_window_snapshot_tracks_partial_coverage_error_and_empty_recovery(tensor):
    block, _ = _make_block(
        responses=[[_model_segment("walk", 0, 1)], RuntimeError("failed"), []],
        tensor=tensor,
    )
    color = {"tensor_rgb_color": [1, 2, 3]} if tensor else {}
    results = [_run(block, _make_frame(n, **color)) for n in range(8)]
    assert [r["window"]["status"] for r in results] == [
        "collecting",
        "collecting",
        "ready",
        "ready",
        "error",
        "error",
        "ready",
        "ready",
    ]
    # The first call has a partial model window: do not infer its start from
    # the configured 1s window or the 0.5s stride. Samples were frames 0 and 2.
    assert results[2]["window"] == {
        "status": "ready",
        "classes": ["walk"],
        "start_frame": 0,
        "end_frame": 2,
        "fps": 4.0,
        "video_identifier": "stream-0",
    }
    assert results[3]["window"] == results[2]["window"]
    assert results[5]["error_status"] == ""  # Existing transient output is unchanged.
    assert results[5]["window"]["start_frame"] is None
    assert results[5]["window"]["end_frame"] is None
    assert results[5]["window"]["classes"] == []
    assert results[6]["window"]["classes"] == []  # Ready-empty differs from error.
    assert (results[6]["window"]["start_frame"], results[6]["window"]["end_frame"]) == (
        4,
        6,
    )
    # Consumers must not be able to mutate the model's held classes.
    results[2]["window"]["classes"].append("injected")
    assert results[3]["window"]["classes"] == ["walk"]


def test_window_uses_actual_sample_end_and_only_declared_fps():
    block, _ = _make_block()
    results = [
        _run(block, _make_frame(n, fps=None, measured_fps=100)) for n in range(16)
    ]
    # Inference uses the 30 FPS fallback internally; visualization must not
    # present that assumption as the source clock, nor use measured throughput.
    assert results[-1]["window"]["status"] == "ready"
    assert results[-1]["window"]["fps"] is None
    block, _ = _make_block()
    for n in range(4):
        result = _run(block, _make_frame(n), stride_seconds=0.75)
    assert result["window"]["end_frame"] == 2  # Call frame 3 was not sampled.


def test_window_resets_on_rewind_filter_change_and_other_stream():
    block, _ = _make_block(responses=[[_model_segment("walk")]])
    for n in range(3):
        _run(block, _make_frame(n))
    assert (
        _run(block, _make_frame(3, video_id="other"))["window"]["status"]
        == "collecting"
    )
    assert _run(block, _make_frame(0))["window"]["status"] == "collecting"
    for n in (1, 2):
        _run(block, _make_frame(n))
    assert (
        _run(block, _make_frame(3), class_filter=["run"])["window"]["status"]
        == "collecting"
    )


@pytest.mark.parametrize("mode", ["compact", "timeline"])
def test_action_recognition_workflow_connects_both_visualization_modes(mode):
    from roboflow_workflows.execution_engine.core import ExecutionEngine

    model = _FakeActionRecognitionModel(responses=[[_model_segment("walk", 0, 1)]])
    model.video_sampling = VideoSampling(window_seconds=1, sample_fps=2, min_frames=1)
    provider = MagicMock()
    provider.load_action_recognition_model.return_value = model
    workflow = {
        "version": "1.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": [
            {
                "type": "roboflow_core/roboflow_action_recognition_model@v2",
                "name": "actions",
                "images": "$inputs.image",
                "model_id": "cosmos-3-edge",
                "stride_seconds": 0.5,
            },
            {
                "type": "roboflow_core/action_recognition_visualization@v1",
                "name": "visualize",
                "image": "$inputs.image",
                "window": "$steps.actions.window",
                "mode": mode,
                **(
                    {"timeline": "$steps.actions.timeline"}
                    if mode == "timeline"
                    else {}
                ),
            },
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "image",
                "selector": "$steps.visualize.image",
            },
            {
                "type": "JsonField",
                "name": "window",
                "selector": "$steps.actions.window",
            },
        ],
    }
    engine = ExecutionEngine.init(
        workflow_definition=workflow,
        init_parameters={
            "workflows_core.model_manager": provider,
            "workflows_core.api_key": None,
            "workflows_core.step_execution_mode": StepExecutionMode.LOCAL,
        },
    )
    for frame in range(4):
        small = _make_frame(frame)
        image = WorkflowImageData.copy_and_replace(
            origin_image_data=small, numpy_image=np.zeros((180, 320, 3), dtype=np.uint8)
        )
        result = engine.run(runtime_parameters={"image": image})[0]
    assert len(model.calls) == 1
    assert result["window"]["classes"] == ["walk"]
    assert result["window"]["end_frame"] == 2
    assert result["image"].video_metadata.frame_number == 3
    assert result["image"].numpy_image.shape == (180, 320, 3)
    assert np.any(result["image"].numpy_image)


@pytest.mark.parametrize("tensor", [False, True])
def test_v2_preserves_v1_timeline_and_error_contract_and_tensor_sampling(tensor):
    responses = [[_model_segment("walk", 0, 1)], RuntimeError("failed"), []]
    v1, _ = _make_v1_block(responses=responses, tensor=tensor)
    v2, model = _make_block(responses=responses, tensor=tensor)
    assert {o.name for o in v1.get_manifest().describe_outputs()} == {
        "timeline",
        "error_status",
    }
    assert {o.name for o in v2.get_manifest().describe_outputs()} == {
        "timeline",
        "error_status",
        "latest_predictions",
        "window",
    }
    color = {"tensor_rgb_color": [1, 2, 3]} if tensor else {}
    for n in range(8):
        old = _run(v1, _make_frame(n, **color))
        new = _run(v2, _make_frame(n, **color))
        assert set(old) == {"timeline", "error_status"}
        assert old == {key: new[key] for key in old}
    assert isinstance(
        model.calls[0]["frames"][0], torch.Tensor if tensor else np.ndarray
    )


def test_both_versions_remain_registered():
    from roboflow_workflows.core_steps.loader import load_blocks

    manifests = {
        b.get_manifest().model_fields["type"].annotation.__args__[0]: b.get_manifest()
        for b in load_blocks()
    }
    assert {
        o.name
        for o in manifests[
            "roboflow_core/roboflow_action_recognition_model@v1"
        ].describe_outputs()
    } == {"timeline", "error_status"}
    assert {
        o.name
        for o in manifests[
            "roboflow_core/roboflow_action_recognition_model@v2"
        ].describe_outputs()
    } == {"timeline", "error_status", "latest_predictions", "window"}
