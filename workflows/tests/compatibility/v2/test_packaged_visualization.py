"""Canonical compiled painters retain independent pixel and ownership references.

Accurate boxes use supervision's oracle; explicit fast boxes retain the V1
band raster oracle. Labels use supervision in both modes. Prototype-only
prepared drawing, fused batch and in-place mechanics are archived separately.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
import supervision as sv
import torch
from pydantic import Field
from roboflow_workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    gpu_draw_boxes,
)
from roboflow_workflows.v2.blocks import visualization
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import OBJECT_DETECTION_PREDICTION_KIND
from roboflow_workflows.v2.engine import (
    Block,
    BlockParams,
    Catalogue,
    CompileOptions,
    Output,
    Ref,
    compile_workflow,
)
from roboflow_workflows.v2.engine.pipelining import PipelineOptions

from inference_models.models.base.object_detection import Detections

CLASS_NAMES = {0: "person", 1: "car", 2: "dog", 5: "bus"}
PALETTE = sv.ColorPalette.DEFAULT
PAINTER_TYPES = {
    "boxes": "roboflow_core/bounding_box_visualization@v1",
    "labels": "roboflow_core/label_visualization@v2",
}


class FakeDetector(Block):
    """Emits the caller's ``Detections`` for each image id, as one batch."""

    type = "test/fake_detector@v1"
    outputs = {
        "predictions": Output(
            OBJECT_DETECTION_PREDICTION_KIND,
            source="image",
            description="Fixed detections of this image.",
        )
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to 'detect' on."
        )

    def __init__(self, *, detections_by_image: Dict[str, Detections]):
        self._by_image = detections_by_image

    def run(self, *, image: Batch) -> List[Dict[str, Any]]:
        results = [{"predictions": self._by_image[one.image_id]} for one in image]

        return results


def _catalogue() -> Catalogue:
    catalogue = Catalogue((FakeDetector, *visualization.BLOCKS))

    return catalogue


def _detections(
    xyxy: Sequence[Sequence[float]],
    class_id: Sequence[int],
    confidence: Sequence[float],
    *,
    class_names: Optional[Dict[int, str]] = CLASS_NAMES,
) -> Detections:
    metadata = {} if class_names is None else {"class_names": dict(class_names)}
    detections = Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor(class_id, dtype=torch.int64),
        confidence=torch.tensor(confidence, dtype=torch.float32),
        image_metadata=metadata,
    )

    return detections


def _scene_detections() -> Detections:
    # Overlapping boxes, a box crossing the top-left corner, one crossing the
    # bottom-right corner: labels clipped at frame edges and later-wins order.
    detections = _detections(
        [[10, 30, 80, 90], [50, 40, 150, 115], [-5, -5, 30, 20], [140, 100, 170, 130]],
        [0, 1, 2, 5],
        [0.91, 0.5, 0.333, 0.777],
    )

    return detections


def _image(
    *, image_id: str, height: int = 120, width: int = 160, seed: int = 0
) -> ImageData:
    generator = torch.Generator().manual_seed(seed)
    pixels = torch.randint(
        0, 256, (3, height, width), generator=generator, dtype=torch.uint8
    )
    image = ImageData.from_tensor(pixels, image_id=image_id)

    return image


def _definition(steps: List[Dict[str, Any]], output: str) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": [
            {
                "type": "test/fake_detector@v1",
                "name": "detector",
                "image": "$inputs.image",
            },
            *steps,
        ],
        "outputs": [{"type": "JsonField", "name": "annotated", "selector": output}],
    }

    return definition


def _painter_step(
    block_type: str,
    *,
    name: str = "painter",
    image: str = "$inputs.image",
    predictions: str = "$steps.detector.predictions",
) -> Dict[str, Any]:
    step = {
        "type": block_type,
        "name": name,
        "image": image,
        "predictions": predictions,
    }

    return step


def _reference_labels(pixels: torch.Tensor, detections: Detections) -> torch.Tensor:
    if len(detections) == 0:
        return pixels.clone()

    names = detections.image_metadata["class_names"]
    labels = [
        f"{names[int(one)]} {score:.2f}"
        for one, score in zip(detections.class_id, detections.confidence.numpy())
    ]
    annotator = sv.LabelAnnotator(
        color=PALETTE,
        text_color=sv.Color.WHITE,
        text_scale=1.0,
        text_thickness=1,
        text_padding=10,
        text_position=sv.Position.TOP_LEFT,
        color_lookup=sv.ColorLookup.CLASS,
        border_radius=0,
    )
    bgr = pixels.permute(1, 2, 0).numpy()[..., ::-1].copy()
    supervision_detections = sv.Detections(
        xyxy=detections.xyxy.numpy(),
        class_id=detections.class_id.numpy(),
        confidence=detections.confidence.numpy(),
    )
    bgr = annotator.annotate(bgr, supervision_detections, labels=labels)
    annotated = torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1)

    return annotated


def _steps(painters):
    steps = []
    if painters in ("boxes", "both"):
        steps.append(_painter_step(PAINTER_TYPES["boxes"], name="boxes"))
    if painters in ("labels", "both"):
        step = _painter_step(
            PAINTER_TYPES["labels"],
            name="labels",
            image="$steps.boxes.image" if painters == "both" else "$inputs.image",
        )
        step["text"] = "Class and Confidence"
        steps.append(step)
    return steps, "$steps." + steps[-1]["name"] + ".image"


def _plan(painters, *, quality="accurate"):
    steps, output = _steps(painters)
    return compile_workflow(
        _definition(steps, output),
        catalogue=_catalogue(),
        options=CompileOptions(quality=quality),
    )


def _run(painters, images, detections, *, quality="accurate"):
    with _plan(painters, quality=quality).create_session(
        {"detections_by_image": detections}
    ) as session:
        result = [row["annotated"] for row in session.run({"image": images}).rows()]
    return result


def _reference_boxes(pixels, detections, *, quality):
    if not len(detections):
        return pixels.clone()
    if quality == "fast":
        scene = pixels.clone()
        colors = np.asarray(
            [PALETTE.by_idx(int(one)).as_rgb() for one in detections.class_id],
            dtype=np.uint8,
        )
        gpu_draw_boxes(scene, detections.xyxy.numpy().astype(int), colors, 2)
        return scene
    bgr = pixels.permute(1, 2, 0).numpy()[..., ::-1].copy()
    boxes = sv.Detections(
        xyxy=detections.xyxy.numpy(),
        class_id=detections.class_id.numpy(),
        confidence=detections.confidence.numpy(),
    )
    bgr = sv.BoxAnnotator(
        color=PALETTE, thickness=2, color_lookup=sv.ColorLookup.CLASS
    ).annotate(bgr, boxes)
    return torch.from_numpy(bgr[..., ::-1].copy()).permute(2, 0, 1)


def _reference(pixels, detections, painters, *, quality):
    scene = pixels
    if painters in ("boxes", "both"):
        scene = _reference_boxes(scene, detections, quality=quality)
    if painters in ("labels", "both"):
        scene = _reference_labels(scene, detections)
    return scene


@pytest.mark.parametrize("quality", ["accurate", "fast"])
@pytest.mark.parametrize("painters", ["boxes", "labels", "both"])
def test_copy_painters_match_independent_references(painters, quality):
    images = [_image(image_id="a", seed=1), _image(image_id="b", seed=2)]
    detections = {"a": _scene_detections(), "b": _detections([], [], [])}
    before = [image.tensor_image.clone() for image in images]
    annotated = _run(painters, images, detections, quality=quality)
    for image, original, output in zip(images, before, annotated):
        expected = _reference(
            image.tensor_image, detections[image.image_id], painters, quality=quality
        )
        assert torch.equal(output.tensor_image, expected)
        assert torch.equal(image.tensor_image, original)
        assert output.tensor_image.data_ptr() != image.tensor_image.data_ptr()
        assert output.image_id == image.image_id
    assert not torch.equal(annotated[0].tensor_image, images[0].tensor_image)


@pytest.mark.parametrize("quality", ["accurate", "fast"])
def test_pipelined_runs_paint_every_image_like_serial_runs(quality):
    batches = [
        [
            _image(image_id=f"{batch}-{index}", seed=10 * batch + index)
            for index in range(2)
        ]
        for batch in range(4)
    ]
    detections = {
        image.image_id: _scene_detections() for batch in batches for image in batch
    }
    plan = _plan("both", quality=quality)
    with plan.create_session({"detections_by_image": detections}) as session:
        with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
            futures = [pipeline.submit({"image": batch}) for batch in batches]
        results = [future.result(timeout=60).rows() for future in futures]
    for batch, rows in zip(batches, results):
        for image, row in zip(batch, rows):
            expected = _reference(
                image.tensor_image, _scene_detections(), "both", quality=quality
            )
            assert row["annotated"].image_id == image.image_id
            assert torch.equal(row["annotated"].tensor_image, expected)


@pytest.mark.parametrize("quality", ["accurate", "fast"])
def test_copy_painters_keep_inputs_geometry_and_metadata(quality):
    root = _image(image_id="frame", height=120, width=160)
    crop = root.crop((20, 10, 140, 110), image_id="crop")
    detections = {"crop": _scene_detections()}
    before = crop.tensor_image.clone()
    prediction_before = detections["crop"].xyxy.clone()
    (output,) = _run("both", [crop], detections, quality=quality)
    assert torch.equal(crop.tensor_image, before)
    assert torch.equal(detections["crop"].xyxy, prediction_before)
    assert output.image_id == crop.image_id
    assert output.parent == crop.parent
    assert output.root == crop.root
    assert output.size_hw == crop.size_hw
    assert output.video_metadata == crop.video_metadata
    assert output.prediction_metadata() == crop.prediction_metadata()
    assert output.tensor_image.is_contiguous()


@pytest.mark.parametrize("quality", ["accurate", "fast"])
def test_strided_inputs_are_painted_like_contiguous_ones(quality):
    contiguous = _image(image_id="a", seed=7)
    channels_last = (
        contiguous.tensor_image.permute(1, 2, 0).contiguous().permute(2, 0, 1)
    )
    wider = torch.zeros((3, 120, 200), dtype=torch.uint8)
    wider[:, :, 20:180] = contiguous.tensor_image
    sliced = wider[:, :, 20:180]
    assert not channels_last.is_contiguous() and not sliced.is_contiguous()
    images = [
        ImageData.from_tensor(channels_last, image_id="b"),
        ImageData.from_tensor(sliced, image_id="c"),
    ]
    detections = {key: _scene_detections() for key in "bc"}
    expected = _reference(
        contiguous.tensor_image, _scene_detections(), "both", quality=quality
    )
    for output in _run("both", images, detections, quality=quality):
        assert torch.equal(output.tensor_image, expected)
    assert torch.equal(sliced, contiguous.tensor_image)


class _FakeEvent:
    synchronized = 0
    recorded = []

    def record(self, stream) -> None:
        self.stream = stream
        _FakeEvent.recorded.append(stream)

    def synchronize(self) -> None:
        _FakeEvent.synchronized += 1


class _FakeStreamContext:
    def __init__(self, stream) -> None:
        self.stream = stream

    def __enter__(self):
        return self.stream

    def __exit__(self, *exc_info) -> None:
        return None


def test_ready_scope_waits_once_per_call_also_on_failure(monkeypatch) -> None:
    from roboflow_workflows.v2.blocks.visualization.annotations.label_visualization._sprites import (
        _ReadyScope,
    )

    created: List[Any] = []

    dependencies = []
    caller_stream = object()

    class SideStream:
        def wait_stream(self, other):
            dependencies.append(other)

    def make_stream(device):
        created.append(device)
        return SideStream()

    monkeypatch.setattr(torch.cuda, "Stream", make_stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: caller_stream)
    monkeypatch.setattr(torch.cuda, "stream", _FakeStreamContext)
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    _FakeEvent.synchronized = 0
    _FakeEvent.recorded = []
    scope = _ReadyScope()
    device = torch.device("cuda", 0)

    with scope.work(device):
        pass
    with pytest.raises(RuntimeError, match="kernel failed"):
        with scope.work(device):
            raise RuntimeError("kernel failed")

    assert _FakeEvent.synchronized == 2
    assert len(_FakeEvent.recorded) == 2
    assert _FakeEvent.recorded[0] is _FakeEvent.recorded[1]
    assert len(created) == 1
    assert dependencies == [caller_stream, caller_stream]
