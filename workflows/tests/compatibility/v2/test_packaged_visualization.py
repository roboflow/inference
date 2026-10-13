"""Packaged batched detection painters (``v2.blocks.visualization``).

Pixel references are independent of the package: boxes come from the V1
``gpu_draw_boxes`` primitive called directly, labels from supervision's own
``LabelAnnotator`` (the V1 tensor label path is exact to it). CUDA tests at
the end are skipped without a GPU.

A fake detector block emits fixed native ``Detections`` per image id, so no
model is loaded.
"""

import subprocess
import sys
from pathlib import Path
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
from roboflow_workflows.v2.blocks.visualization import blocks as painter_blocks
from roboflow_workflows.v2.blocks.visualization import streams
from roboflow_workflows.v2.blocks.visualization.drawing import (
    PALETTE,
    DetectionDrawing,
    prepare_drawings,
)
from roboflow_workflows.v2.blocks.visualization.kernels import (
    TensorKernels,
)
from roboflow_workflows.v2.blocks.visualization.labels import (
    SPRITE_CACHE_SIZE,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.v2.engine.errors import (
    ParamsValidationError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.pipelining import PipelineOptions
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.object_detection import Detections
from tests.v2.support.paths import WORKFLOWS_ROOT

CLASS_NAMES = {0: "person", 1: "car", 2: "dog", 5: "bus"}
WORKFLOWS_DIR = WORKFLOWS_ROOT


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
    catalogue = Catalogue(
        (FakeDetector, *visualization.BLOCKS), kinds=list(visualization.KINDS)
    )

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


def _run(
    definition: Dict[str, Any],
    images: List[ImageData],
    detections_by_image: Dict[str, Detections],
    *,
    mutation_conflicts: str = "error",
) -> List[Any]:
    plan = compile_workflow(
        definition,
        catalogue=_catalogue(),
        options=CompileOptions(mutation_conflicts=mutation_conflicts),
    )
    session = plan.create_session({"detections_by_image": detections_by_image})
    annotated = [row["annotated"] for row in session.run({"image": images}).rows()]

    return annotated


# --- independent references -------------------------------------------------


def _reference_boxes(pixels: torch.Tensor, detections: Detections) -> torch.Tensor:
    scene = pixels.clone()
    if len(detections) == 0:
        return scene

    colors = np.asarray(
        [PALETTE.by_idx(int(one)).as_rgb() for one in detections.class_id],
        dtype=np.uint8,
    )
    gpu_draw_boxes(scene, detections.xyxy.numpy().astype(int), colors, 2)

    return scene


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


def _reference(
    pixels: torch.Tensor, detections: Detections, painters: str
) -> torch.Tensor:
    scene = pixels
    if painters in ("boxes", "both"):
        scene = _reference_boxes(scene, detections)
    if painters in ("labels", "both"):
        scene = _reference_labels(scene, detections)

    return scene


# --- pixels through compiled workflows ---------------------------------------

PAINTER_TYPES = {
    "boxes": "v2/box_visualization@v1",
    "labels": "v2/label_visualization@v1",
    "both": "v2/detection_visualization@v1",
}


@pytest.mark.parametrize("painters", ["boxes", "labels", "both"])
def test_copy_painters_match_independent_references(painters: str) -> None:
    images = [_image(image_id="a", seed=1), _image(image_id="b", seed=2)]
    detections = {"a": _scene_detections(), "b": _detections([], [], [])}

    annotated = _run(
        _definition([_painter_step(PAINTER_TYPES[painters])], "$steps.painter.image"),
        images,
        detections,
    )

    for image, output in zip(images, annotated):
        expected = _reference(image.tensor_image, detections[image.image_id], painters)
        assert torch.equal(output.tensor_image, expected)
    assert not torch.equal(annotated[0].tensor_image, images[0].tensor_image)


def test_routes_with_shared_preparation_and_in_place_labels_match_fused() -> None:
    images = [_image(image_id="a", seed=3), _image(image_id="b", seed=4)]
    detections = {"a": _scene_detections(), "b": _scene_detections()}
    fused = _run(
        _definition([_painter_step(PAINTER_TYPES["both"])], "$steps.painter.image"),
        images,
        detections,
    )
    shared = _run(
        _definition(
            [
                {
                    "type": "v2/detection_drawing@v1",
                    "name": "drawing",
                    "predictions": "$steps.detector.predictions",
                },
                _painter_step(
                    PAINTER_TYPES["boxes"],
                    name="boxes",
                    predictions="$steps.drawing.drawing",
                ),
                _painter_step(
                    "v2/label_visualization_in_place@v1",
                    name="labels",
                    image="$steps.boxes.image",
                    predictions="$steps.drawing.drawing",
                ),
            ],
            "$steps.labels.image",
        ),
        images,
        detections,
    )

    for one, other in zip(fused, shared):
        assert torch.equal(one.tensor_image, other.tensor_image)


def test_pipelined_runs_paint_every_batch_like_serial_runs() -> None:
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
    definition = _definition(
        [
            {
                "type": "v2/detection_drawing@v1",
                "name": "drawing",
                "predictions": "$steps.detector.predictions",
            },
            _painter_step(
                PAINTER_TYPES["boxes"],
                name="boxes",
                predictions="$steps.drawing.drawing",
            ),
            _painter_step(
                "v2/label_visualization_in_place@v1",
                name="labels",
                image="$steps.boxes.image",
                predictions="$steps.drawing.drawing",
            ),
        ],
        "$steps.labels.image",
    )
    plan = compile_workflow(
        definition,
        catalogue=_catalogue(),
        options=CompileOptions(mutation_conflicts="error"),
    )
    session = plan.create_session({"detections_by_image": detections})

    with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
        futures = [pipeline.submit({"image": batch}) for batch in batches]
    results = [future.result(timeout=60).rows() for future in futures]

    for batch, rows in zip(batches, results):
        for image, row in zip(batch, rows):
            expected = _reference(image.tensor_image, _scene_detections(), "both")
            assert row["annotated"].image_id == image.image_id
            assert torch.equal(row["annotated"].tensor_image, expected)


def test_painters_given_prepared_drawings_make_no_transfer(monkeypatch) -> None:
    transferred: List[int] = []
    original = painter_blocks.prepare_drawings

    def recording(predictions):
        transferred.append(len(predictions))
        return original(predictions)

    monkeypatch.setattr(painter_blocks, "prepare_drawings", recording)
    images = [_image(image_id="a"), _image(image_id="b"), _image(image_id="c")]
    detections = {key: _scene_detections() for key in "abc"}

    _run(
        _definition(
            [
                {
                    "type": "v2/detection_drawing@v1",
                    "name": "drawing",
                    "predictions": "$steps.detector.predictions",
                },
                _painter_step(
                    PAINTER_TYPES["boxes"],
                    name="boxes",
                    predictions="$steps.drawing.drawing",
                ),
                _painter_step(
                    PAINTER_TYPES["labels"],
                    name="labels",
                    image="$steps.boxes.image",
                    predictions="$steps.drawing.drawing",
                ),
            ],
            "$steps.labels.image",
        ),
        images,
        detections,
    )

    # The prep block transfers the batch of 3 once; both painters receive
    # prepared drawings and have nothing native left to transfer.
    assert transferred == [3, 0, 0]


def test_one_call_per_batch_and_results_stay_aligned(monkeypatch) -> None:
    calls: List[int] = []
    original = painter_blocks._Painter._paint_copies

    def counting(self, image, predictions, **kwargs):
        calls.append(len(image))
        return original(self, image, predictions, **kwargs)

    monkeypatch.setattr(painter_blocks._Painter, "_paint_copies", counting)
    sizes = [(40, 60), (90, 50), (64, 64)]
    images = [
        _image(image_id=f"i{index}", height=height, width=width, seed=index)
        for index, (height, width) in enumerate(sizes)
    ]
    detections = {
        "i0": _detections([[5, 30, 30, 39]], [1], [0.6]),
        "i1": _detections([], [], []),
        "i2": _detections([[0, 25, 63, 63], [10, 40, 20, 50]], [0, 2], [0.2, 0.99]),
    }

    annotated = _run(
        _definition([_painter_step(PAINTER_TYPES["both"])], "$steps.painter.image"),
        images,
        detections,
    )

    assert calls == [3]
    for image, output in zip(images, annotated):
        expected = _reference(image.tensor_image, detections[image.image_id], "both")
        assert output.image_id == image.image_id
        assert torch.equal(output.tensor_image, expected)


def test_empty_detections_give_a_new_unchanged_copy() -> None:
    image = _image(image_id="a")

    (output,) = _run(
        _definition([_painter_step(PAINTER_TYPES["both"])], "$steps.painter.image"),
        [image],
        {"a": _detections([], [], [], class_names=None)},
    )

    assert torch.equal(output.tensor_image, image.tensor_image)
    assert output.tensor_image.data_ptr() != image.tensor_image.data_ptr()


def test_copy_painters_keep_inputs_geometry_and_metadata() -> None:
    root = _image(image_id="frame", height=120, width=160)
    crop = root.crop((20, 10, 140, 110), image_id="crop")
    detections = {"crop": _scene_detections()}
    before = crop.tensor_image.clone()
    prediction_before = detections["crop"].xyxy.clone()

    (output,) = _run(
        _definition([_painter_step(PAINTER_TYPES["both"])], "$steps.painter.image"),
        [crop],
        detections,
    )

    assert torch.equal(crop.tensor_image, before)
    assert torch.equal(detections["crop"].xyxy, prediction_before)
    assert output.image_id == crop.image_id
    assert output.parent == crop.parent
    assert output.root == crop.root
    assert output.size_hw == crop.size_hw
    assert output.video_metadata == crop.video_metadata
    assert output.prediction_metadata() == crop.prediction_metadata()
    assert output.tensor_image.is_contiguous()


def test_strided_inputs_are_painted_like_contiguous_ones() -> None:
    contiguous = _image(image_id="a", seed=7)
    # Channels-last storage and a column slice of a wider frame: both strided.
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
    expected = _reference(contiguous.tensor_image, _scene_detections(), "both")

    annotated = _run(
        _definition([_painter_step(PAINTER_TYPES["both"])], "$steps.painter.image"),
        images,
        detections,
    )

    for output in annotated:
        assert torch.equal(output.tensor_image, expected)
    assert torch.equal(sliced, contiguous.tensor_image)


def test_prepared_drawing_is_exact_read_only_and_reads_strided_columns() -> None:
    # Model-style output: one (N, 6) tensor; class_id is a strided column,
    # including the length-1 case PyTorch calls contiguous whatever its stride.
    big_id = 2**53 + 1
    raw = torch.tensor(
        [[1.5, 2.5, 30.25, 40.75, 0.875, 0.0], [5.0, 6.0, 7.0, 8.0, 0.5, 0.0]],
        dtype=torch.float64,
    )
    class_ids = torch.tensor([[big_id], [3]], dtype=torch.int64)[:, 0]
    many = Detections(
        xyxy=raw[:, :4],
        class_id=class_ids,
        confidence=raw[:, 4],
        image_metadata={"class_names": {big_id: "rare", 3: "cat"}},
    )
    single = Detections(
        xyxy=raw[:1, :4],
        class_id=raw[:1, 5].to(torch.int64),
        confidence=raw[:1, 4],
        image_metadata={"class_names": ["person"]},
    )

    first, second = painter_blocks.DetectionDrawingBlock().run(
        predictions=[many, single]
    )

    drawing = first["drawing"]
    assert isinstance(drawing, DetectionDrawing)
    assert drawing.xyxy.dtype == np.float64
    assert np.array_equal(drawing.xyxy, raw[:, :4].numpy())
    assert drawing.class_id.tolist() == [big_id, 3]
    assert drawing.labels == ("rare 0.88", "cat 0.50")
    assert second["drawing"].labels == ("person 0.88",)
    with pytest.raises(ValueError, match="read-only"):
        drawing.xyxy[0, 0] = 0.0


def test_in_place_painter_changes_its_input_and_refuses_strided_images() -> None:
    image = _image(image_id="a")
    pixels = image.tensor_image
    expected = _reference_labels(pixels, _scene_detections())
    painter = painter_blocks.InPlaceLabelVisualizationBlock()

    (result,) = painter.run(image=[image], predictions=[_scene_detections()])

    assert result["image"] is image
    assert torch.equal(pixels, expected)

    strided = ImageData.from_tensor(pixels.permute(0, 2, 1), image_id="t")
    with pytest.raises(ValueError, match="contiguous"):
        painter.run(image=[strided], predictions=[_scene_detections()])


def test_in_place_painter_declares_its_mutation_to_the_compiler() -> None:
    steps = [
        _painter_step(PAINTER_TYPES["boxes"], name="boxes"),
        _painter_step(
            "v2/label_visualization_in_place@v1",
            name="labels",
            image="$steps.boxes.image",
        ),
        # An unordered second reader of the box output would see the labels.
        _painter_step(
            PAINTER_TYPES["labels"], name="reader", image="$steps.boxes.image"
        ),
    ]

    assert painter_blocks.InPlaceLabelVisualizationBlock.mutates == ("image",)
    with pytest.raises(Exception, match="mutated in place"):
        compile_workflow(
            _definition(steps, "$steps.labels.image"),
            catalogue=_catalogue(),
            options=CompileOptions(mutation_conflicts="error"),
        )


# --- invalid configuration and inputs ----------------------------------------


@pytest.mark.parametrize(
    "painter_type", [*PAINTER_TYPES.values(), "v2/label_visualization_in_place@v1"]
)
@pytest.mark.parametrize("backend", ["tensor", "triton"])
def test_obsolete_backend_field_fails_with_a_useful_message(
    painter_type: str, backend: str
) -> None:
    # Definitions written before the field was removed must not compile and
    # silently draw another way.
    step = {**_painter_step(painter_type), "backend": backend}

    with pytest.raises(ParamsValidationError, match="no longer a painter") as raised:
        compile_workflow(
            _definition([step], "$steps.painter.image"), catalogue=_catalogue()
        )

    assert raised.value.field_path == ("backend",)
    assert "catalogue" in str(raised.value)


@pytest.mark.parametrize(
    "detections, message",
    [
        (_detections([[1, 1, 9, 9]], [0], [0.5], class_names=None), "class_names"),
        (_detections([[1, 1, 9, 9]], [7], [0.5]), "class id 7"),
    ],
)
def test_missing_class_names_fail_before_painting(detections, message) -> None:
    image = _image(image_id="a")
    before = image.tensor_image.clone()

    with pytest.raises(StepExecutionError, match=message):
        _run(
            _definition([_painter_step(PAINTER_TYPES["both"])], "$steps.painter.image"),
            [image],
            {"a": detections},
        )

    assert torch.equal(image.tensor_image, before)


def test_negative_class_id_fails_before_in_place_painting() -> None:
    images = [_image(image_id="a"), _image(image_id="b")]
    before = [one.tensor_image.clone() for one in images]
    drawings = prepare_drawings(
        [
            _scene_detections(),
            _detections([[1, 1, 9, 9]], [-1], [0.5], class_names={-1: "x"}),
        ]
    )

    with pytest.raises(ValueError, match="negative class id"):
        painter_blocks.InPlaceLabelVisualizationBlock().run(
            image=images, predictions=drawings
        )

    for image, pixels in zip(images, before):
        assert torch.equal(image.tensor_image, pixels)


def test_misaligned_batches_and_mixed_inputs_are_rejected() -> None:
    painter = painter_blocks.DetectionVisualizationBlock()
    gray = ImageData.from_tensor(
        torch.zeros((1, 8, 8), dtype=torch.uint8), image_id="g"
    )

    with pytest.raises(ValueError, match="must align"):
        painter.run(image=[_image(image_id="a")], predictions=[])
    with pytest.raises(ValueError, match="RGB"):
        painter.run(image=[gray], predictions=[_detections([], [], [])])
    assert painter.run(image=[], predictions=[]) == []


def test_painters_paint_without_importing_triton() -> None:
    # Triton is optional for torch; importing or running the painters must not
    # need it. An import of it fails in this subprocess.
    script = """
import sys
sys.modules["triton"] = None
import torch
from roboflow_workflows.v2.blocks import visualization
from roboflow_workflows.v2.data.image import ImageData
from inference_models.models.base.object_detection import Detections
image = ImageData.from_tensor(torch.zeros((3, 32, 32), dtype=torch.uint8), image_id="a")
detections = Detections(xyxy=torch.tensor([[2.0, 12.0, 20.0, 30.0]]),
    class_id=torch.tensor([0]), confidence=torch.tensor([0.5]),
    image_metadata={"class_names": {0: "person"}})
result = visualization.DetectionVisualizationBlock().run(image=[image], predictions=[detections])
assert result[0]["image"].tensor_image.sum().item() > 0
print("painted")
"""
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        capture_output=True,
        text=True,
        check=True,
        cwd=WORKFLOWS_DIR.parent,
    )

    assert completed.stdout.strip().splitlines()[-1] == "painted"


# --- bounded caches ------------------------------------------------------------


def test_label_sprite_cache_stays_bounded() -> None:
    count = SPRITE_CACHE_SIZE + 40
    names = {index: f"class{index}" for index in range(count)}
    detections = _detections(
        [[10 + index % 50, 40, 60, 80] for index in range(count)],
        list(range(count)),
        [0.5] * count,
        class_names=names,
    )
    (drawing,) = prepare_drawings([detections])
    scene = torch.zeros((3, 120, 200), dtype=torch.uint8)

    kernels = TensorKernels()
    kernels.paint([scene], [drawing], boxes=False, labels=True)

    assert kernels.placement.cached_sprites == SPRITE_CACHE_SIZE


# --- readiness -------------------------------------------------------------------


class _FakeEvent:
    synchronized = 0

    def record(self, stream) -> None:
        self.stream = stream

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
    scope = streams.ReadyScope()
    device = torch.device("cuda", 0)

    with scope.work(device):
        pass
    with pytest.raises(RuntimeError, match="kernel failed"):
        with scope.work(device):
            raise RuntimeError("kernel failed")

    assert _FakeEvent.synchronized == 2
    assert len(created) == 1
    assert dependencies == [caller_stream, caller_stream]


def test_changing_frame_widths_keeps_sprite_templates_bounded() -> None:
    kernels = TensorKernels()
    drawings = prepare_drawings([_detections([[100, 100, 150, 140]], [0], [0.9])])
    for width in range(400, 432):
        scene = torch.zeros((3, 200, width), dtype=torch.uint8)
        kernels.paint([scene], drawings, boxes=False, labels=True)

    assert kernels.placement.cached_sprites <= SPRITE_CACHE_SIZE
    assert all(
        len(sprite._flat_by_width) == 1
        for sprite in kernels.placement._sprites.values()
    )


# --- CUDA (skipped without a GPU; run on Thor) -------------------------------------

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@cuda
def test_cuda_painters_match_cpu_pixels() -> None:
    images = [_image(image_id="a", seed=21), _image(image_id="b", height=71, width=97)]
    detections = [_scene_detections(), _scene_detections()]
    expected = [
        _reference(one.tensor_image, det, "both")
        for one, det in zip(images, detections)
    ]
    cuda_images = [one.with_pixels(one.tensor_image.cuda()) for one in images]
    cuda_detections = [
        Detections(
            xyxy=det.xyxy.cuda(),
            class_id=det.class_id.cuda(),
            confidence=det.confidence.cuda(),
            image_metadata=det.image_metadata,
        )
        for det in detections
    ]

    results = painter_blocks.DetectionVisualizationBlock().run(
        image=cuda_images, predictions=cuda_detections
    )

    for result, reference in zip(results, expected):
        assert result["image"].tensor_image.is_cuda
        assert torch.equal(result["image"].tensor_image.cpu(), reference)


@cuda
@pytest.mark.parametrize("mode", ["run", "phases"])
def test_cuda_painter_follows_pending_image_transform(mode: str) -> None:
    catalogue = Catalogue.from_modules(["roboflow_workflows.v2.blocks"])
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "image", "kind": ["image"]},
            {
                "type": "WorkflowParameter",
                "name": "predictions",
                "kind": [OBJECT_DETECTION_PREDICTION_KIND.name],
            },
        ],
        "steps": [
            {"type": "v2/invert", "name": "invert", "image": "$inputs.image"},
            {
                "type": "v2/detection_visualization@v1",
                "name": "paint",
                "image": "$steps.invert.image",
                "predictions": "$inputs.predictions",
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "image", "selector": "$steps.paint.image"}
        ],
    }
    plan = compile_workflow(
        definition, catalogue=catalogue, options=CompileOptions(block_execution=mode)
    )
    pixels = torch.zeros((3, 512, 512), dtype=torch.uint8, device="cuda")
    image = ImageData.from_tensor(pixels)
    detections = _detections([], [], [])
    torch.cuda.synchronize()
    with plan.create_session() as session:
        for _ in range(3):
            # Make ordinary upstream work remain pending when painting starts.
            torch.cuda._sleep(100_000_000)
            result = session.run({"image": image, "predictions": detections})
            painted = result.rows()[0]["image"].tensor_image
            assert torch.all(painted == 255).item()
