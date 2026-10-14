"""Packaged ``roboflow_core/roboflow_object_detection_model@v3`` through compiled workflows.

A fake model with the native ``inference_models`` API (``pre_process``,
``forward``, ``post_process``, ``class_names``) stands in for loaded weights,
so nothing is downloaded. It records every call to prove one model call per
phase per batch.
"""

import datetime
from typing import Any, Dict, List

import pytest
import torch
from roboflow_workflows.execution_engine.entities.base import VideoMetadata
from roboflow_workflows.v2.blocks.models.roboflow_platform.object_detection_model.v3 import (
    ObjectDetectionBlock,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import FRAME_RECORD_KEY, FrameRef, frame_of
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.data.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import row_record, without_row_record
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.errors import (
    KindMismatchError,
    ParamsValidationError,
    ResourceError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.introspection import discover_workload
from roboflow_workflows.v2.engine.phases import PhaseFailure
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.object_detection import Detections

MODES = ("run", "phases")
MODEL_ID = "fake-od"
# network_input options of a step that sets only the default confidence.
UNSET_OPTIONS = {
    "confidence": 0.4,
    "iou_threshold": None,
    "max_detections": None,
    "class_agnostic_nms": None,
    "class_filter": None,
}


class FakeDetectionModel:
    """Fixed boxes per image; ``extra_predictions`` breaks the count contract."""

    class_names = ["person", "car", "dog"]

    def __init__(self, *, extra_predictions: int = 0):
        self.extra_predictions = extra_predictions
        self.calls: List[str] = []
        self.batch_sizes: List[int] = []
        self.thresholds: List[float] = []

    def pre_process(self, images: List[torch.Tensor], input_color_format: str):
        assert input_color_format == "rgb"
        self.calls.append("pre_process")
        self.batch_sizes.append(len(images))
        # One brightness per image: images of one batch may differ in size.
        batch = torch.stack([image.float().mean() / 255.0 for image in images])
        metadata = [tuple(image.shape[1:]) for image in images]

        return batch, metadata

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        self.calls.append("forward")

        return batch.clone()

    def post_process(self, raw: torch.Tensor, metadata: list, confidence: float):
        self.calls.append("post_process")
        self.thresholds.append(confidence)
        predictions = []
        for brightness, (height, width) in zip(raw.tolist(), metadata):
            # Box geometry depends on the image size; confidence on brightness,
            # so each image of a batch gets its own predictions.
            xyxy = torch.tensor(
                [[1.0, 2.0, width / 2, height / 2], [0.0, 0.0, width, height]]
            )
            scores = torch.tensor([0.9, round(brightness, 3)])
            keep = scores >= confidence
            predictions.append(
                Detections(
                    xyxy=xyxy[keep],
                    class_id=torch.tensor([0, 2])[keep],
                    confidence=scores[keep],
                    image_metadata={"model_note": "kept"},
                )
            )
        predictions.extend(predictions[:1] * self.extra_predictions)

        return predictions


def _catalogue() -> Catalogue:
    catalogue = Catalogue(
        [ObjectDetectionBlock],
        kinds=[IMAGE_KIND, OBJECT_DETECTION_PREDICTION_KIND, FLOAT_KIND],
    )

    return catalogue


def _definition(*, batch_input: bool, model_id: str = MODEL_ID) -> Dict[str, Any]:
    image_input = "WorkflowBatchInput" if batch_input else "WorkflowParameter"
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": image_input, "name": "image", "kind": ["image"]},
            {
                "type": "WorkflowParameter",
                "name": "confidence",
                "kind": ["float"],
                "default_value": 0.4,
            },
        ],
        "steps": [
            {
                "type": "roboflow_core/roboflow_object_detection_model@v3",
                "name": "detector",
                "image": "$inputs.image",
                "model_id": model_id,
                "confidence": "$inputs.confidence",
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "predictions",
                "selector": "$steps.detector.predictions",
            }
        ],
    }

    return definition


def _session(
    models: Dict[str, Any],
    *,
    batch_input: bool,
    mode: str = "run",
    model_id: str = MODEL_ID,
):
    plan = compile_workflow(
        _definition(batch_input=batch_input, model_id=model_id),
        catalogue=_catalogue(),
        options=CompileOptions(block_execution=mode),
    )
    session = plan.create_session({"models": models})

    return session


def _image(
    height: int = 40, width: int = 60, *, value: int = 128, image_id: str = "frame"
) -> ImageData:
    pixels = torch.full((3, height, width), value, dtype=torch.uint8)
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _predictions(rows: List[Dict[str, Any]]) -> List[Detections]:
    predictions = [row["predictions"] for row in rows]

    return predictions


@pytest.mark.parametrize("mode", MODES)
def test_single_image_is_a_batch_of_one_with_one_call_per_phase(mode: str) -> None:
    model = FakeDetectionModel()
    session = _session({MODEL_ID: model}, batch_input=False, mode=mode)

    (row,) = session.run({"image": _image()}).rows()

    assert model.calls == ["pre_process", "forward", "post_process"]
    assert model.batch_sizes == [1]
    assert isinstance(row["predictions"], Detections)


@pytest.mark.parametrize("mode", MODES)
def test_batch_input_runs_each_model_phase_once_and_keeps_input_order(
    mode: str,
) -> None:
    model = FakeDetectionModel()
    session = _session({MODEL_ID: model}, batch_input=True, mode=mode)
    images = [
        _image(40, 60, value=200, image_id="a"),
        _image(30, 50, value=20, image_id="b"),
        _image(20, 80, value=150, image_id="c"),
    ]

    rows = session.run({"image": images, "confidence": 0.5}).rows()

    assert model.calls == ["pre_process", "forward", "post_process"]
    assert model.batch_sizes == [3]
    predictions = _predictions(rows)
    assert [p.image_metadata["parent_id"] for p in predictions] == ["a", "b", "c"]
    # Image "b" is dark: its second box falls below the threshold.
    assert [len(p) for p in predictions] == [2, 1, 2]
    assert predictions[1].xyxy.tolist() == [[1.0, 2.0, 25.0, 15.0]]


@pytest.mark.parametrize("mode", MODES)
def test_wrong_prediction_count_fails_the_step(mode: str) -> None:
    model = FakeDetectionModel(extra_predictions=1)
    session = _session({MODEL_ID: model}, batch_input=True, mode=mode)

    with pytest.raises(StepExecutionError, match="3 predictions for 2 images"):
        session.run({"image": [_image(image_id="a"), _image(image_id="b")]})


def test_confidence_selector_reaches_post_process() -> None:
    model = FakeDetectionModel()
    session = _session({MODEL_ID: model}, batch_input=False)

    low = session.run({"image": _image(value=100), "confidence": 0.1}).rows()
    high = session.run({"image": _image(value=100), "confidence": 0.95}).rows()

    assert model.thresholds == [0.1, 0.95]
    assert len(low[0]["predictions"]) == 2
    assert len(high[0]["predictions"]) == 0


def test_confidence_outside_unit_interval_is_rejected() -> None:
    model = FakeDetectionModel()
    session = _session({MODEL_ID: model}, batch_input=False)

    with pytest.raises(StepExecutionError, match="confidence"):
        session.run({"image": _image(), "confidence": 1.5})

    assert model.calls == []


def test_phases_and_run_give_identical_predictions() -> None:
    images = [_image(value=200, image_id="a"), _image(30, 70, value=90, image_id="b")]

    by_mode = {}
    for mode in MODES:
        session = _session(
            {MODEL_ID: FakeDetectionModel()}, batch_input=True, mode=mode
        )
        by_mode[mode] = _predictions(session.run({"image": images}).rows())

    for run_prediction, phase_prediction in zip(by_mode["run"], by_mode["phases"]):
        for field in ("xyxy", "class_id", "confidence"):
            assert torch.equal(
                getattr(run_prediction, field), getattr(phase_prediction, field)
            )
        # Each result mints its own row namespace; everything else is equal.
        assert without_row_record(run_prediction.image_metadata) == (
            without_row_record(phase_prediction.image_metadata)
        )
        run_rows, phase_rows = row_record(run_prediction), row_record(phase_prediction)
        assert (run_rows.rows, run_rows.ids) == (phase_rows.rows, phase_rows.ids)
        assert run_rows.namespace != phase_rows.namespace


def test_predictions_keep_model_metadata_class_names_and_image_frame() -> None:
    video = VideoMetadata(
        video_identifier="camera-3",
        frame_number=41,
        frame_timestamp=datetime.datetime(
            2026, 10, 12, 8, 30, tzinfo=datetime.timezone.utc
        ),
        fps=25.0,
    )
    pixels = torch.full((3, 200, 300), 128, dtype=torch.uint8)
    frame = ImageData.from_tensor(pixels, image_id="frame", video_metadata=video)
    crop = frame.crop([40, 20, 240, 180], image_id="crop")
    block = ObjectDetectionBlock(models={MODEL_ID: FakeDetectionModel()})

    (result,) = block.run(image=Batch([crop]), model_id=MODEL_ID, confidence=0.4)

    metadata = result["predictions"].image_metadata
    assert metadata["model_note"] == "kept"
    assert metadata["class_names"] == {0: "person", 1: "car", 2: "dog"}
    assert metadata["prediction_type"] == "object-detection"
    assert metadata["parent_id"] == "crop"
    assert metadata["root_parent_id"] == "frame"
    assert metadata["image_dimensions"] == [160, 200]
    assert metadata["root_parent_coordinates"] == [40, 20]
    assert result["predictions"].xyxy.tolist()[0] == [1.0, 2.0, 100.0, 80.0]
    assert metadata[FRAME_RECORD_KEY] == FrameRef.from_image(crop).to_record()
    predicted_frame = frame_of(result["predictions"])
    assert predicted_frame.image_id == "crop"
    assert predicted_frame.size_hw == (160, 200)
    assert predicted_frame.root.frame_id == "frame"
    assert predicted_frame.root.frame_size_hw == (200, 300)
    assert predicted_frame.root.offset_xy == (40.0, 20.0)
    assert predicted_frame.root.scale_xy == (1.0, 1.0)
    assert predicted_frame.video_metadata == video
    assert predicted_frame.video_metadata.video_identifier == "camera-3"
    assert predicted_frame.video_metadata.frame_number == 41


def test_absent_model_id_fails_before_any_model_call() -> None:
    model = FakeDetectionModel()
    session = _session({"other": model}, batch_input=False, model_id="missing")

    with pytest.raises(StepExecutionError) as caught:
        session.run({"image": _image()})

    cause = caught.value.__cause__
    while cause is not None and not isinstance(cause, LookupError):
        cause = cause.__cause__
    assert isinstance(cause, LookupError)
    assert "'missing'" in str(cause) and "['other']" in str(cause)
    assert model.calls == []


def test_session_without_models_resource_fails_at_construction() -> None:
    plan = compile_workflow(
        _definition(batch_input=False),
        catalogue=_catalogue(),
        options=CompileOptions(block_execution="run"),
    )

    with pytest.raises(ResourceError, match="models"):
        plan.create_session({}).run({"image": _image()})


def test_non_mapping_models_resource_is_rejected() -> None:
    with pytest.raises(TypeError, match="mapping of model id"):
        ObjectDetectionBlock(models=[FakeDetectionModel()])


def test_model_id_must_not_be_empty() -> None:
    definition = _definition(batch_input=False, model_id="")

    with pytest.raises(ParamsValidationError, match="model_id"):
        compile_workflow(definition, catalogue=_catalogue())


def test_model_id_selector_must_select_a_string() -> None:
    definition = _definition(batch_input=False, model_id="$inputs.confidence")

    with pytest.raises(KindMismatchError, match="model_id accepts kinds"):
        compile_workflow(definition, catalogue=_catalogue())


def test_host_mapping_changes_do_not_rebind_a_running_block() -> None:
    first, second = FakeDetectionModel(), FakeDetectionModel()
    models = {MODEL_ID: first}
    session = _session(models, batch_input=False)
    session.run({"image": _image()})

    models[MODEL_ID] = second
    del models[MODEL_ID]
    models["late"] = second
    session.run({"image": _image()})

    assert first.calls == ["pre_process", "forward", "post_process"] * 2
    assert second.calls == []


def test_mapping_snapshot_keeps_the_model_objects_themselves() -> None:
    model = FakeDetectionModel()
    models = {MODEL_ID: model}

    block = ObjectDetectionBlock(models=models)
    models.clear()
    network_input = block.network_input(
        Batch([_image()]), MODEL_ID, None, **UNSET_OPTIONS
    )

    assert network_input.model is model


def test_new_session_with_new_mapping_binds_the_new_model() -> None:
    first, second = FakeDetectionModel(), FakeDetectionModel()
    plan = compile_workflow(
        _definition(batch_input=False),
        catalogue=_catalogue(),
        options=CompileOptions(block_execution="run"),
    )

    plan.create_session({"models": {MODEL_ID: first}}).run({"image": _image()})
    plan.create_session({"models": {MODEL_ID: second}}).run({"image": _image()})

    assert len(first.calls) == len(second.calls) == 3


def test_every_phase_uses_the_model_resolved_by_network_input() -> None:
    first, second = FakeDetectionModel(), FakeDetectionModel()
    block = ObjectDetectionBlock(models={"first": first, "second": second})
    images = Batch([_image()])

    network_input = block.network_input(images, "first", None, **UNSET_OPTIONS)
    raw_output = block.raw_output(network_input)
    block.result(raw_output, network_input, images)

    assert first.calls == ["pre_process", "forward", "post_process"]
    assert second.calls == []


def test_non_rgb_image_is_rejected_before_the_model_runs() -> None:
    model = FakeDetectionModel()
    gray = ImageData.from_tensor(torch.zeros((1, 10, 10), dtype=torch.uint8))
    block = ObjectDetectionBlock(models={MODEL_ID: model})

    with pytest.raises(PhaseFailure, match="3-channel RGB"):
        block.run(image=Batch([gray]), model_id=MODEL_ID, confidence=0.4)

    assert model.calls == []


def test_discovery_reports_the_literal_model_id() -> None:
    plan = compile_workflow(_definition(batch_input=False), catalogue=_catalogue())

    workload = discover_workload(plan)

    resources = [
        (item.resource_type, item.identifier, item.used_by_steps)
        for item in workload.resources.items
    ]
    assert resources == [("model", MODEL_ID, ["$steps.detector"])]
    assert workload.resources.complete
    (declared,) = ObjectDetectionBlock.discover_dependent_resources(
        ObjectDetectionBlock.Params(image="$inputs.image", model_id=MODEL_ID)
    )
    assert declared.details == {
        "resource": "models",
        "task_type": "object-detection",
        "model_package_id": None,
    }
