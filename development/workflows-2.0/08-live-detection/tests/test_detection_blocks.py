"""Tests of the live detection blocks through the compiled workflow.

A fake model with the native ``inference_models`` API replaces the platform
weights; the painters are the real V1 tensor primitives. Run from the
repository root::

    PYTHONPATH=.:workflows:inference_models:stream_vision python -m pytest \
        development/workflows-2.0/08-live-detection/tests
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.append(str(EXAMPLE_DIR))

import detection_blocks  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import supervision as sv  # noqa: E402
import torch  # noqa: E402
from detection_blocks import (  # noqa: E402
    PALETTE,
    BoxVisualization,
    LabelVisualization,
    ObjectDetector,
    create_catalogue,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import (  # noqa: E402
    ImageData,
)
from roboflow_workflows.execution_engine.v2.compilation import (  # noqa: E402
    compile_workflow,
)
from roboflow_workflows.execution_engine.v2.errors import (  # noqa: E402
    StepExecutionError,
)
from roboflow_workflows.execution_engine.v2.plan import CompileOptions  # noqa: E402

from inference_models.models.base.object_detection import Detections  # noqa: E402

WORKFLOW = json.loads((EXAMPLE_DIR / "workflows" / "live_detection.json").read_text())
MODES = ("run", "phases")
TIMINGS = ("pre_ms", "model_ms", "post_ms", "boxes_ms", "labels_ms")


class FakeDetectionModel:
    """Native ``pre_process`` / ``forward`` / ``post_process`` with fixed boxes.

    Records what the detector passes in. ``batch_size`` predictions are
    returned per call, to test the one-image contract.
    """

    class_names = ["person", "car", "dog"]

    def __init__(
        self,
        xyxy: List[List[int]],
        class_id: List[int],
        confidence: List[float],
        *,
        batch_size: int = 1,
    ):
        self.xyxy = torch.tensor(xyxy, dtype=torch.int32).reshape(-1, 4)
        self.class_id = torch.tensor(class_id, dtype=torch.int32)
        self.confidence = torch.tensor(confidence, dtype=torch.float32)
        self.batch_size = batch_size
        self.pre_process_calls: List[Dict[str, Any]] = []
        self.thresholds: List[float] = []

    def pre_process(self, images: torch.Tensor, input_color_format: Optional[str]):
        self.pre_process_calls.append(
            {"pixels": images.clone(), "color_format": input_color_format}
        )
        batch = images.unsqueeze(0).float() / 255.0

        return batch, [{"original_size": tuple(images.shape[1:])}]

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return batch.mean(dim=(2, 3))

    def post_process(self, raw: torch.Tensor, metadata: list, confidence: float):
        self.thresholds.append(confidence)
        keep = self.confidence >= confidence
        predictions = [
            Detections(
                xyxy=self.xyxy[keep],
                class_id=self.class_id[keep],
                confidence=self.confidence[keep],
                image_metadata={"model_note": "kept"},
            )
            for _ in range(self.batch_size)
        ]

        return predictions


def _model(**overrides: Any) -> FakeDetectionModel:
    arguments = {
        "xyxy": [[20, 60, 90, 110], [120, 70, 180, 115]],
        "class_id": [0, 2],
        "confidence": [0.91, 0.55],
    }
    arguments.update(overrides)
    model = FakeDetectionModel(**arguments)

    return model


def _image(height: int = 120, width: int = 200, *, image_id: str = "frame"):
    # Asymmetric channels: red ramps along x, green along y, blue constant.
    xs = torch.linspace(0, 255, width).to(torch.uint8).expand(height, width)
    ys = torch.linspace(0, 255, height).to(torch.uint8)[:, None].expand(height, width)
    blue = torch.full((height, width), 40, dtype=torch.uint8)
    pixels = torch.stack([xs, ys, blue]).contiguous()
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _run(
    model: FakeDetectionModel,
    image: ImageData,
    *,
    mode: str = "run",
    confidence: float = 0.4,
) -> Dict[str, Any]:
    plan = compile_workflow(
        WORKFLOW,
        catalogue=create_catalogue(),
        options=CompileOptions(block_execution=mode, mutation_conflicts="error"),
    )
    session = plan.create_session({"detection_model": model})
    (row,) = session.run({"image": image, "confidence": confidence}).rows()

    return row


def _to_bgr(pixels: torch.Tensor) -> np.ndarray:
    bgr = np.ascontiguousarray(pixels.permute(1, 2, 0).numpy()[:, :, ::-1])

    return bgr


def test_phases_and_run_give_same_predictions_pixels_and_timings() -> None:
    image = _image()

    by_mode = {mode: _run(_model(), image, mode=mode) for mode in MODES}

    run_row, phases_row = by_mode["run"], by_mode["phases"]
    for name in ("xyxy", "class_id", "confidence"):
        assert torch.equal(
            getattr(run_row["predictions"], name),
            getattr(phases_row["predictions"], name),
        )
    assert (
        run_row["predictions"].image_metadata
        == phases_row["predictions"].image_metadata
    )
    assert torch.equal(
        run_row["annotated"].tensor_image, phases_row["annotated"].tensor_image
    )
    for row in by_mode.values():
        for name in TIMINGS:
            assert isinstance(row[name], float) and row[name] >= 0.0


def test_detector_passes_rgb_tensor_and_confidence_to_the_model() -> None:
    image = _image()
    model = _model()

    row = _run(model, image, confidence=0.6)

    (call,) = model.pre_process_calls
    assert call["color_format"] == "rgb"
    assert torch.equal(call["pixels"], image.tensor_image)
    assert model.thresholds == [0.6]
    assert row["predictions"].class_id.tolist() == [0]


def _crop_and_resize() -> ImageData:
    frame = _image(200, 300, image_id="frame")
    crop = frame.crop([40, 20, 240, 180], image_id="crop")
    small = crop.resize((80, 100), image_id="small")

    return small


def test_detector_predictions_carry_crop_and_resize_provenance() -> None:
    small = _crop_and_resize()
    model = _model(xyxy=[[5, 5, 50, 40]], class_id=[1], confidence=[0.8])

    result = ObjectDetector(detection_model=model).run(image=small, confidence=0.4)

    metadata = result["predictions"].image_metadata
    assert metadata["parent_id"] == "small"
    assert metadata["parent_frame_id"] == "crop"
    assert metadata["root_parent_id"] == "frame"
    assert metadata["image_dimensions"] == [80, 100]
    assert metadata["parent_dimensions"] == [160, 200]
    assert metadata["root_parent_dimensions"] == [200, 300]
    assert metadata["parent_coordinates"] == [0, 0]
    assert metadata["root_parent_coordinates"] == [40, 20]
    assert metadata["scaling_relative_to_parent"] == 0.5
    assert metadata["scaling_relative_to_root_parent"] == 0.5
    assert metadata["class_names"] == {0: "person", 1: "car", 2: "dog"}
    assert metadata["prediction_type"] == "object-detection"
    assert metadata["model_note"] == "kept"


def test_workflow_outputs_keep_image_identity_and_map_boxes_to_the_input_root():
    small = _crop_and_resize()
    model = _model(xyxy=[[5, 5, 50, 40]], class_id=[1], confidence=[0.8])

    row = _run(model, small)

    # The engine's default output coordinate system is the workflow root
    # (V1 parity): root_xy = local_xy / 0.5 + (40, 20).
    predictions = row["predictions"]
    assert predictions.xyxy.tolist() == [[50, 30, 140, 100]]
    assert predictions.image_metadata["parent_id"] == "frame"
    assert predictions.image_metadata["image_dimensions"] == [200, 300]
    assert predictions.image_metadata["class_names"][1] == "car"
    annotated = row["annotated"]
    assert annotated.image_id == "small"
    assert annotated.parent == small.parent
    assert annotated.root == small.root


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("batch_size", [0, 2])
def test_detector_rejects_other_than_one_prediction(mode: str, batch_size: int):
    model = _model(batch_size=batch_size)

    with pytest.raises(StepExecutionError) as error:
        _run(model, _image(), mode=mode)

    assert isinstance(error.value.__cause__, ValueError)
    assert f"returned {batch_size} predictions" in str(error.value.__cause__)


def test_confidence_outside_unit_interval_is_rejected() -> None:
    with pytest.raises(StepExecutionError, match="confidence"):
        _run(_model(), _image(), confidence=1.5)


@pytest.mark.parametrize("mode", MODES)
def test_empty_detections_give_an_unchanged_owned_copy(mode: str) -> None:
    image = _image()

    row = _run(_model(xyxy=[], class_id=[], confidence=[]), image, mode=mode)

    assert len(row["predictions"]) == 0
    assert "class_names" in row["predictions"].image_metadata
    annotated = row["annotated"].tensor_image
    assert torch.equal(annotated, image.tensor_image)
    assert annotated.data_ptr() != image.tensor_image.data_ptr()


@pytest.mark.parametrize("mode", MODES)
def test_source_pixels_are_not_modified(mode: str) -> None:
    image = _image()
    original = image.tensor_image.clone()

    row = _run(_model(), image, mode=mode)

    assert torch.equal(image.tensor_image, original)
    assert not torch.equal(row["annotated"].tensor_image, original)


def test_non_contiguous_input_gives_contiguous_annotated_image() -> None:
    contiguous = _image()
    hwc = np.ascontiguousarray(contiguous.tensor_image.permute(1, 2, 0).numpy())
    view = torch.from_numpy(hwc).permute(2, 0, 1)
    assert not view.is_contiguous()

    row = _run(_model(), ImageData.from_tensor(view))
    reference = _run(_model(), contiguous)

    assert row["annotated"].tensor_image.is_contiguous()
    assert torch.equal(
        row["annotated"].tensor_image, reference["annotated"].tensor_image
    )


def test_labels_refuse_a_non_contiguous_image() -> None:
    hwc = np.zeros((40, 60, 3), dtype=np.uint8)
    image = ImageData.from_tensor(torch.from_numpy(hwc).permute(2, 0, 1))
    predictions = _model().post_process(None, [], confidence=0.0)[0]

    with pytest.raises(ValueError, match="contiguous"):
        LabelVisualization().run(image=image, predictions=predictions)


def test_boxes_are_painted_in_rgb_class_colors_on_the_border_only() -> None:
    image = ImageData.from_tensor(torch.zeros(3, 120, 200, dtype=torch.uint8))
    model = _model(xyxy=[[120, 70, 180, 115]], class_id=[2], confidence=[0.9])

    row = _run(model, image)

    pixels = row["annotated"].tensor_image
    expected = torch.tensor(PALETTE.by_idx(2).as_rgb(), dtype=torch.uint8)
    assert torch.equal(pixels[:, 90, 120], expected)
    assert torch.equal(pixels[:, 90, 180], expected)
    assert torch.equal(pixels[:, 115, 150], expected)
    assert int(pixels[:, 90, 150].sum()) == 0


def test_labels_match_supervision_label_annotator_exactly() -> None:
    # Second box sits at the top edge: its label crosses the frame border
    # and takes the clipped sprite path.
    model = _model(
        xyxy=[[20, 60, 90, 110], [120, 4, 190, 60]],
        class_id=[0, 2],
        confidence=[0.91, 0.55],
    )
    image = _image()

    row = _run(model, image, confidence=0.0)

    detections = sv.Detections(
        xyxy=model.xyxy.numpy().astype(float),
        class_id=model.class_id.numpy().astype(int),
        confidence=model.confidence.numpy(),
    )
    boxes_bgr = _to_bgr(_boxes_only(model, image))
    annotator = sv.LabelAnnotator(
        color=PALETTE,
        color_lookup=sv.ColorLookup.CLASS,
        text_position=sv.Position.TOP_LEFT,
        text_color=sv.Color.WHITE,
        text_scale=1.0,
        text_thickness=1,
        text_padding=10,
        border_radius=0,
    )
    expected = annotator.annotate(
        scene=boxes_bgr, detections=detections, labels=["person 0.91", "dog 0.55"]
    )
    assert np.array_equal(_to_bgr(row["annotated"].tensor_image), expected)


def test_sprite_cache_stays_within_its_bound(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(detection_blocks, "SPRITE_CACHE_SIZE", 3)
    model = _model(
        xyxy=[[10 + 30 * i, 60, 35 + 30 * i, 100] for i in range(5)],
        class_id=[0] * 5,
        confidence=[0.51, 0.52, 0.53, 0.54, 0.55],
    )
    labels = LabelVisualization()
    image = _image()
    predictions = model.post_process(None, [], confidence=0.0)[0]
    predictions.image_metadata = {"class_names": {0: "person"}}

    labels.run(image=image, predictions=predictions)

    assert len(labels._sprites) == 3


def test_labels_refuse_predictions_without_class_names() -> None:
    predictions = _model().post_process(None, [], confidence=0.0)[0]

    with pytest.raises(ValueError, match="class_names"):
        LabelVisualization().run(image=_image(), predictions=predictions)


def test_workflow_compiles_without_mutation_conflicts() -> None:
    for mode in MODES:
        plan = compile_workflow(
            WORKFLOW,
            catalogue=create_catalogue(),
            options=CompileOptions(block_execution=mode, mutation_conflicts="error"),
        )

        assert plan.warnings == ()


def _boxes_only(model: FakeDetectionModel, image: ImageData) -> torch.Tensor:
    predictions = model.post_process(None, [], confidence=0.0)[0]
    boxed = BoxVisualization().run(image=image, predictions=predictions)

    return boxed["image"].tensor_image
