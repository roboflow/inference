"""Current-frame action predictions connect to unchanged classification consumers."""

import importlib
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from roboflow_workflows.core_steps.common.entities import StepExecutionMode
from roboflow_workflows.core_steps.common.serializers_tensor import (
    serialise_native_classification,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    VideoMetadata,
    WorkflowImageData,
)
from roboflow_workflows.execution_engine.v1.compiler.reference_type_checker import (
    validate_reference_kinds,
)

from inference_models.models.base.action_recognition import (
    ActionRecognitionPrediction,
    VideoSampling,
)
from inference_models.models.base.classification import (
    MultiLabelClassificationPrediction,
)


@pytest.fixture(params=["v1", "v1_tensor"])
def variant(request):
    module = importlib.import_module(
        "roboflow_workflows.core_steps.models.roboflow.action_recognition."
        + request.param
    )
    visualizer_module = importlib.import_module(
        "roboflow_workflows.core_steps.visualizations.classification_label."
        + request.param
    )
    return module, visualizer_module


def _frame(number, *, video_id="video", tensor=False):
    pixels = np.zeros((192, 256, 3), dtype=np.uint8)
    image = (
        {"tensor_image": torch.from_numpy(pixels).permute(2, 0, 1)}
        if tensor
        else {"numpy_image": pixels}
    )
    frame = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id=f"{video_id}:{number}"),
        workflow_root_ancestor_metadata=ImageParentMetadata(parent_id="root"),
        video_metadata=VideoMetadata(
            video_identifier=video_id,
            frame_number=number,
            frame_timestamp=datetime(2026, 1, 1),
            fps=4.0,
        ),
        **image,
    )
    return frame


def _block(module, responses, *, class_names=("idle", "walk", "run")):
    model = SimpleNamespace(
        class_names=list(class_names) if class_names is not None else None,
        video_sampling=VideoSampling(window_seconds=4, sample_fps=4, min_frames=1),
        infer=Mock(side_effect=responses),
    )
    provider = Mock()
    provider.load_action_recognition_model.return_value = model
    block = module.ActionRecognitionModelBlockV1(
        api_key=None,
        step_execution_mode=StepExecutionMode.LOCAL,
        model_manager=provider,
    )
    return block, model


def _run(block, image):
    result = block.run(images=[image], model_id="actions/1", stride_seconds=0.5)
    return result[0]


def _serialized(result):
    prediction = result["frame_predictions"]
    if isinstance(prediction, MultiLabelClassificationPrediction):
        prediction = serialise_native_classification(prediction)
    return prediction


def _observe(block, *, tensor=False):
    for number in range(3):
        image = _frame(number, tensor=tensor)
        result = _run(block, image)
    return image, result


def test_frame_predictions_declare_the_visualizers_classification_kind(variant):
    module, visualizer_module = variant
    manifest = module.ActionRecognitionModelBlockV1.get_manifest()
    outputs = {output.name: output for output in manifest.describe_outputs()}
    schema = visualizer_module.ClassificationLabelManifest.model_json_schema()
    expected_kind = schema["properties"]["predictions"]["kind"]

    validate_reference_kinds(
        expected=[kind["name"] for kind in expected_kind],
        actual=outputs["frame_predictions"].kind,
        error_message="Action predictions must connect to classification visualization",
    )
    assert outputs["frame_predictions"].kind[0].internal_data_type == (
        expected_kind[0]["internal_data_type"]
    )


def test_overlapping_actions_use_model_ids_and_synthetic_confidence(variant):
    module, _ = variant
    block, model = _block(
        module,
        [
            [
                ActionRecognitionPrediction(0, 2, "walk"),
                ActionRecognitionPrediction(2, 2, "run"),
                ActionRecognitionPrediction(1, 2, "walk"),
            ]
        ],
    )
    _, result = _observe(block, tensor=module.__name__.endswith("_tensor"))

    assert _serialized(result) == {
        "image": {"height": 192, "width": 256},
        "predictions": {
            "walk": {"class_id": 1, "confidence": 1.0},
            "run": {"class_id": 2, "confidence": 1.0},
        },
        "predicted_classes": ["walk", "run"],
        "prediction_type": "classification",
        "parent_id": "video:2",
        "root_parent_id": "root",
    }
    assert model.infer.call_count == 1
    assert len(result["timeline"]) == 2


@pytest.mark.parametrize("end_frame", [1, 2])
def test_frame_coverage_is_inclusive_and_labels_expire_between_calls(
    variant, end_frame
):
    module, _ = variant
    block, model = _block(module, [[ActionRecognitionPrediction(0, end_frame, "run")]])
    _, result = _observe(block)

    expected_classes = ["run"] if end_frame == 2 else []
    assert _serialized(result)["predicted_classes"] == expected_classes

    next_result = _run(block, _frame(3))
    assert _serialized(next_result)["predictions"] == {}
    assert _serialized(next_result)["predicted_classes"] == []
    assert next_result["timeline"] == result["timeline"]
    assert model.infer.call_count == 1


@pytest.mark.parametrize("response", [[], RuntimeError("model unavailable")])
def test_warmup_empty_results_and_failures_produce_empty_classifications(
    variant, response
):
    module, _ = variant
    block, _ = _block(module, [response])

    warmup = _run(block, _frame(0))
    assert _serialized(warmup)["predictions"] == {}
    assert _serialized(warmup)["image"] == {"height": 192, "width": 256}

    _run(block, _frame(1))
    result = _run(block, _frame(2))
    assert _serialized(result)["predicted_classes"] == []
    assert result["error_status"] == (
        "model unavailable" if isinstance(response, Exception) else ""
    )


def test_vocabulary_free_classes_get_stable_classification_ids(variant):
    module, _ = variant
    block, _ = _block(
        module,
        [
            [ActionRecognitionPrediction(2, 2, "walk")],
            [ActionRecognitionPrediction(4, 4, "run")],
            [ActionRecognitionPrediction(6, 6, "walk")],
        ],
        class_names=None,
    )
    predictions = []
    for number in range(7):
        result = _run(block, _frame(number))
        if number in (2, 4, 6):
            predictions.append(_serialized(result)["predictions"])

    assert predictions == [
        {"walk": {"class_id": 0, "confidence": 1.0}},
        {"run": {"class_id": 1, "confidence": 1.0}},
        {"walk": {"class_id": 0, "confidence": 1.0}},
    ]
    assert all(action.class_id == -1 for action in result["timeline"])


@pytest.mark.parametrize("text", ["Class", "Confidence", "Class and Confidence"])
@pytest.mark.parametrize("matching_action", [True, False])
def test_frame_predictions_render_with_the_unchanged_visualizer(
    variant, text, matching_action
):
    module, visualizer_module = variant
    actions = [ActionRecognitionPrediction(0, 2, "run")] if matching_action else []
    block, _ = _block(module, [actions])
    image, result = _observe(block)
    visualizer = visualizer_module.ClassificationLabelVisualizationBlockV1()
    manifest = visualizer_module.ClassificationLabelManifest(
        type="roboflow_core/classification_label_visualization@v1",
        name="labels",
        image="$inputs.image",
        predictions="$steps.actions.frame_predictions",
        text=text,
    )
    configuration = manifest.model_dump(
        exclude={"type", "name", "image", "predictions"}
    )

    output = visualizer.run(
        image=image, predictions=result["frame_predictions"], **configuration
    )

    assert output["image"].numpy_image.shape == (192, 256, 3)
    assert bool(np.any(output["image"].numpy_image)) is matching_action
    assert not np.any(image.numpy_image)


def test_classification_ids_are_independent_per_stream_and_reset_on_rewind(variant):
    module, _ = variant
    block, _ = _block(
        module,
        [
            [ActionRecognitionPrediction(2, 2, "walk")],
            [ActionRecognitionPrediction(2, 2, "run")],
            [ActionRecognitionPrediction(2, 2, "jump")],
        ],
        class_names=None,
    )
    for number in range(3):
        first = _run(block, _frame(number, video_id="first"))
        second = _run(block, _frame(number, video_id="second"))

    assert _serialized(first)["predictions"] == {
        "walk": {"class_id": 0, "confidence": 1.0}
    }
    assert _serialized(second)["predictions"] == {
        "run": {"class_id": 0, "confidence": 1.0}
    }

    reset = _run(block, _frame(0, video_id="first"))
    assert _serialized(reset)["predicted_classes"] == []
    _run(block, _frame(1, video_id="first"))
    restarted = _run(block, _frame(2, video_id="first"))
    assert _serialized(restarted)["predictions"] == {
        "jump": {"class_id": 0, "confidence": 1.0}
    }


def test_tensor_frame_prediction_metadata_does_not_materialize_the_image():
    from roboflow_workflows.core_steps.models.roboflow.action_recognition import (
        v1_tensor,
    )

    block, _ = _block(v1_tensor, [[ActionRecognitionPrediction(0, 2, "run")]])
    image, result = _observe(block, tensor=True)

    assert isinstance(result["frame_predictions"], MultiLabelClassificationPrediction)
    assert image._numpy_image is None
    assert _serialized(result)["predicted_classes"] == ["run"]
