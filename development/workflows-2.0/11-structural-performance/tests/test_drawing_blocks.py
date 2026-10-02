"""Tests of the shared drawing preparation, its painters and the 11 backend.

The reference is the 09 per-painter pair (``GpuBoxVisualization``,
``GpuLabelVisualization``) on the same detections. Every comparison is exact:
pixels, boxes, class ids and confidences. CPU only. Run from the repository
root::

    PYTHONPATH=.:workflows:inference_models:stream_vision python -m pytest \
        development/workflows-2.0/11-structural-performance/tests
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

import drawing_backend  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import structural_imports  # noqa: E402
import torch  # noqa: E402
from drawing_blocks import (  # noqa: E402
    DETECTION_DRAWING_KIND,
    BatchedDetectionDrawingPrep,
    BatchedDrawingBoxVisualization,
    BatchedDrawingLabelVisualization,
    DetectionDrawing,
    DrawingBoxVisualization,
    DrawingLabelVisualization,
    prepare_drawings,
)
from gpu_blocks import GpuBoxVisualization, GpuLabelVisualization  # noqa: E402
from roboflow_workflows.execution_engine.v2.blocks.image_data import (  # noqa: E402
    ImageData,
)
from roboflow_workflows.execution_engine.v2.data import Batch  # noqa: E402
from roboflow_workflows.execution_engine.v2.errors import ContractError  # noqa: E402
from scripted_model import ScriptedDetectionModel, scripted_detections  # noqa: E402

from inference_models.models.base.object_detection import Detections  # noqa: E402

PARITY_10 = structural_imports.load_10_module("check_parity")
CLASS_NAMES = {index: f"class-{index}" for index in range(80)}
HEIGHT, WIDTH = 240, 320


def _frame(seed: int, *, height: int = HEIGHT, width: int = WIDTH) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    frame = torch.randint(
        0, 256, (3, height, width), dtype=torch.uint8, generator=generator
    )

    return frame


def _detections(
    xyxy: List[List[float]],
    class_id: List[int],
    confidence: List[float],
    *,
    class_names: Dict[int, str] = CLASS_NAMES,
    class_dtype: torch.dtype = torch.int64,
) -> Detections:
    detections = Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor(class_id, dtype=class_dtype),
        confidence=torch.tensor(confidence, dtype=torch.float32),
        image_metadata={"class_names": class_names},
    )

    return detections


def _many(count: int) -> Detections:
    generator = torch.Generator().manual_seed(count)
    corners = torch.rand((count, 2), generator=generator) * torch.tensor(
        [WIDTH, HEIGHT]
    )
    sizes = 10 + torch.rand((count, 2), generator=generator) * 60
    xyxy = torch.cat([corners, corners + sizes], dim=1)
    detections = Detections(
        xyxy=xyxy,
        class_id=torch.randint(0, 80, (count,), generator=generator),
        confidence=torch.rand((count,), generator=generator),
        image_metadata={"class_names": CLASS_NAMES},
    )

    return detections


CASES = {
    "none": lambda: _detections([], [], []),
    "one": lambda: _detections([[40.75, 60.25, 120.5, 140.0]], [3], [0.875]),
    "many_max_detections": lambda: _many(300),
    "offscreen": lambda: _detections(
        [
            [-30.5, -20.25, 40.0, 30.0],  # crosses top-left
            [290.0, 200.0, 360.0, 260.0],  # crosses bottom-right
            [400.0, 300.0, 450.0, 350.0],  # fully outside
            [0.0, 0.0, 319.0, 239.0],  # whole frame
        ],
        [0, 1, 2, 3],
        [0.5, 0.6, 0.7, 0.8],
    ),
    "classes_beyond_palette": lambda: _detections(
        [[10.0 + 12 * k, 30.0 + 7 * k, 60.0 + 12 * k, 90.0 + 7 * k] for k in range(6)],
        [21, 22, 42, 63, 79, 0],
        [0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    ),
    "tied_confidences": lambda: _detections(
        [
            [20.0, 40.0, 80.0, 90.0],
            [20.0, 40.0, 80.0, 90.0],
            [100.0, 40.0, 180.0, 90.0],
        ],
        [5, 5, 7],
        [0.55, 0.55, 0.55],
    ),
    "int32_class_ids": lambda: _detections(
        [[30.0, 50.0, 90.0, 110.0]], [9], [0.25], class_dtype=torch.int32
    ),
    "large_int64_class_ids": lambda: _detections(
        [[30.0, 50.0, 90.0, 110.0], [130.0, 50.0, 190.0, 110.0]],
        [2**53 + 1, 2**62 + 3],
        [0.5, 0.75],
        class_names={2**53 + 1: "big", 2**62 + 3: "huge"},
    ),
}


def _per_painter(image: ImageData, predictions: Detections) -> ImageData:
    boxed = GpuBoxVisualization().run(image=image, predictions=predictions)["image"]
    labelled = GpuLabelVisualization().run(image=boxed, predictions=predictions)

    return labelled["image"]


def _shared(image: ImageData, predictions: Detections) -> ImageData:
    (drawing,) = prepare_drawings([predictions])
    boxed = DrawingBoxVisualization().run(image=image, drawing=drawing)["image"]
    labelled = DrawingLabelVisualization().run(image=boxed, drawing=drawing)

    return labelled["image"]


@pytest.mark.parametrize("case", sorted(CASES))
def test_shared_drawing_paints_the_per_painter_pixels(case: str) -> None:
    predictions = CASES[case]()
    source = _frame(1)
    image = ImageData.from_tensor(source.clone(), image_id="frame")

    expected = _per_painter(image, predictions)
    actual = _shared(image, predictions)

    assert torch.equal(actual.tensor_image, expected.tensor_image)
    assert torch.equal(image.tensor_image, source)  # the input image is untouched
    if case != "none" and case != "offscreen":
        assert not torch.equal(actual.tensor_image, source)  # something was drawn


def test_snapshot_is_exact_detached_and_immutable() -> None:
    predictions = CASES["large_int64_class_ids"]()
    predictions.xyxy[0, 0] = 30.123456789  # not representable in float16
    expected = {
        "xyxy": predictions.xyxy.numpy().copy(),
        "class_id": predictions.class_id.numpy().copy(),
        "confidence": predictions.confidence.numpy().copy(),
    }

    (drawing,) = prepare_drawings([predictions])
    predictions.xyxy.add_(100.0)
    predictions.class_id.zero_()
    predictions.confidence.zero_()

    for name, values in expected.items():
        array = getattr(drawing, name)
        assert array.dtype == values.dtype
        assert np.array_equal(array, values)
        assert array.tobytes() == values.tobytes()  # bit-exact
        assert not array.flags.writeable
        with pytest.raises(ValueError):
            array.setflags(write=True)
    assert drawing.class_id.tolist() == [2**53 + 1, 2**62 + 3]
    assert drawing.labels == ("big 0.50", "huge 0.75")
    with pytest.raises(AttributeError):
        drawing.labels = ()

    (again,) = prepare_drawings([CASES["large_int64_class_ids"]()])
    assert again is not drawing
    assert not np.shares_memory(again.xyxy, drawing.xyxy)


@pytest.mark.parametrize("count", [0, 1, 3])
def test_snapshot_reads_strided_model_columns(count: int) -> None:
    # Real post-processing returns columns of one (N, 6) tensor. With N = 1
    # PyTorch calls such a slice contiguous although its stride is 6.
    table = torch.arange(count * 6, dtype=torch.float32).reshape(count, 6) + 0.5
    predictions = Detections(
        xyxy=table[:, :4],
        class_id=torch.zeros((count, 6), dtype=torch.int64)[:, 5],
        confidence=table[:, 4],
        image_metadata={"class_names": CLASS_NAMES},
    )

    (drawing,) = prepare_drawings([predictions])

    assert np.array_equal(drawing.xyxy, table[:, :4].numpy())
    assert np.array_equal(drawing.confidence, table[:, 4].numpy())
    assert drawing.class_id.tolist() == [0] * count


def test_labels_are_the_08_texts() -> None:
    import detection_blocks

    predictions = _many(50)

    (drawing,) = prepare_drawings([predictions])

    assert list(drawing.labels) == detection_blocks._class_and_confidence_labels(
        predictions
    )


def test_prepare_rejects_what_it_cannot_draw_exactly() -> None:
    mixed = _detections([[1.0, 2.0, 3.0, 4.0]], [0], [0.5])
    mixed.class_id = mixed.class_id.to("meta")
    with pytest.raises(ValueError, match="one device"):
        prepare_drawings([mixed])

    on_cpu = _detections([[1.0, 2.0, 3.0, 4.0]], [0], [0.5])
    elsewhere = Detections(
        xyxy=torch.zeros((1, 4), device="meta"),
        class_id=torch.zeros((1,), dtype=torch.int64, device="meta"),
        confidence=torch.zeros((1,), device="meta"),
    )
    with pytest.raises(ValueError, match="one device"):
        prepare_drawings([on_cpu, elsewhere])

    short = _detections([[1.0, 2.0, 3.0, 4.0]], [0], [0.5])
    short.confidence = torch.zeros((2,))
    with pytest.raises(ValueError, match="confidence"):
        prepare_drawings([short])

    unnamed = _detections([[1.0, 2.0, 3.0, 4.0]], [7], [0.5], class_names={0: "a"})
    with pytest.raises(ValueError, match="no name"):
        prepare_drawings([unnamed])

    no_names = Detections(
        xyxy=torch.zeros((1, 4)),
        class_id=torch.zeros((1,), dtype=torch.int64),
        confidence=torch.zeros((1,)),
    )
    with pytest.raises(ValueError, match="class_names"):
        prepare_drawings([no_names])

    unnamed_but_empty = Detections(
        xyxy=torch.zeros((0, 4)),
        class_id=torch.zeros((0,), dtype=torch.int64),
        confidence=torch.zeros((0,)),
    )
    (empty,) = prepare_drawings([unnamed_but_empty])
    assert len(empty) == 0


def test_kind_accepts_drawings_only() -> None:
    (drawing,) = prepare_drawings([CASES["one"]()])

    DETECTION_DRAWING_KIND.check(drawing)
    with pytest.raises(ContractError):
        DETECTION_DRAWING_KIND.check({"xyxy": drawing.xyxy})
    with pytest.raises(ValueError, match="read-only"):
        DetectionDrawing(
            xyxy=np.zeros((1, 4)),
            class_id=drawing.class_id,
            confidence=drawing.confidence,
            labels=drawing.labels,
        )
    with pytest.raises(ValueError, match="labels"):
        DetectionDrawing(
            xyxy=drawing.xyxy,
            class_id=drawing.class_id,
            confidence=drawing.confidence,
            labels=(),
        )


def test_batched_prep_keeps_order_counts_and_values() -> None:
    members = [
        CASES["many_max_detections"](),
        CASES["none"](),
        CASES["large_int64_class_ids"](),
        CASES["int32_class_ids"](),
        CASES["none"](),
    ]
    one_by_one = [prepare_drawings([member])[0] for member in members]

    results = BatchedDetectionDrawingPrep().run(predictions=Batch(members))

    assert len(results) == len(members)
    assert len({result["prep_ms"] for result in results}) == 1  # one batch time
    for result, expected, member in zip(results, one_by_one, members):
        drawing = result["drawing"]
        assert len(drawing) == len(member)
        assert drawing.labels == expected.labels
        for name in ("xyxy", "class_id", "confidence"):
            assert getattr(drawing, name).tobytes() == getattr(expected, name).tobytes()
            assert getattr(drawing, name).dtype == getattr(expected, name).dtype


def test_batch_painters_match_per_image_painters_on_mixed_sizes() -> None:
    sizes = [(240, 320), (120, 200), (300, 180)]
    sources = [_frame(seed, height=h, width=w) for seed, (h, w) in enumerate(sizes)]
    images = [
        ImageData.from_tensor(source.clone(), image_id=f"f{index}")
        for index, source in enumerate(sources)
    ]
    predictions = [
        scripted_detections(10 + 7 * index, height=h, width=w)
        for index, (h, w) in enumerate(sizes)
    ]
    for detections in predictions:
        detections.image_metadata = {"class_names": CLASS_NAMES}
    drawings = prepare_drawings(predictions)

    boxed = BatchedDrawingBoxVisualization().run(
        image=Batch(images), drawing=Batch(drawings)
    )
    boxed_images = [result["image"] for result in boxed]
    labelled = BatchedDrawingLabelVisualization().run(
        image=Batch(boxed_images), drawing=Batch(drawings)
    )

    for index, result in enumerate(labelled):
        assert result["image"] is boxed_images[index]  # labels paint in place
        assert result["image"].image_id == f"f{index}"
        expected = _per_painter(images[index], predictions[index])
        assert torch.equal(result["image"].tensor_image, expected.tensor_image)
    for image, source in zip(images, sources):
        assert torch.equal(image.tensor_image, source)


def test_label_painter_mutates_only_the_bound_image() -> None:
    source = _frame(3)
    image = ImageData.from_tensor(source.clone(), image_id="frame")
    (drawing,) = prepare_drawings([CASES["classes_beyond_palette"]()])

    boxed = DrawingBoxVisualization().run(image=image, drawing=drawing)["image"]
    boxed_before = boxed.tensor_image.clone()
    labelled = DrawingLabelVisualization().run(image=boxed, drawing=drawing)["image"]

    assert boxed.tensor_image.data_ptr() != image.tensor_image.data_ptr()
    assert labelled is boxed
    assert not torch.equal(labelled.tensor_image, boxed_before)
    assert torch.equal(image.tensor_image, source)
    assert labelled.image_id == image.image_id
    assert labelled.parent == image.parent and labelled.root == image.root


# Workflow level: every drawing, run and phase execution, single and mixed
# batches, against per-painter with session.run.

FRAME_SIZES = ((360, 640), (288, 512), (540, 960))
FRAMES = 12


def _rows(mode: str, drawing: str, *, batch_size: int, frames, ids) -> List[dict]:
    backend = drawing_backend.build_for_model(
        mode,
        model=ScriptedDetectionModel(),
        model_id="scripted",
        confidence=0.4,
        pipeline_depth=2,
        drawing=drawing,
    )
    try:
        plan = PARITY_10.batch_plan(len(frames), batch_size=batch_size)
        rows = PARITY_10.run_plan(backend, frames, ids=ids, plan=plan)
        views = [PARITY_10._comparable(backend, row) for row in rows]
        row_ids = [PARITY_10._row_ids(backend, row) for row in rows]
    finally:
        backend.close()

    return [{"view": view, "ids": seen} for view, seen in zip(views, row_ids)]


@pytest.fixture(scope="module")
def frames_and_reference() -> Dict[str, Any]:
    frames = PARITY_10.marker_frames(FRAMES, sizes_hw=FRAME_SIZES)
    ids = [f"s{index % 4}-f{index}" for index in range(FRAMES)]
    originals = [frame.clone() for frame in frames]
    reference = _rows("v2_serial", "per-painter", batch_size=4, frames=frames, ids=ids)

    return {
        "frames": frames,
        "ids": ids,
        "originals": originals,
        "reference": reference,
    }


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("mode", ["v2_serial", "v2_pipeline"])
@pytest.mark.parametrize("drawing", drawing_backend.DRAWINGS)
def test_workflow_rows_equal_per_painter(
    drawing: str, mode: str, batch_size: int, frames_and_reference: Dict[str, Any]
) -> None:
    frames, ids = frames_and_reference["frames"], frames_and_reference["ids"]

    rows = _rows(mode, drawing, batch_size=batch_size, frames=frames, ids=ids)

    detections = [len(row["view"]["class_id"]) for row in rows]
    assert min(detections) == 0 and max(detections) >= 20  # empty and busy frames
    for row, expected, image_id in zip(rows, frames_and_reference["reference"], ids):
        assert row["ids"] == (image_id, image_id)
        for name, value in expected["view"].items():
            assert torch.equal(row["view"][name], value), name
    for frame, original in zip(frames, frames_and_reference["originals"]):
        assert torch.equal(frame, original)


def test_calls_and_transfers_per_run(monkeypatch: pytest.MonkeyPatch) -> None:
    import batched_blocks
    import drawing_blocks

    counted = {
        "per_painter_boxes": batched_blocks.GpuBoxVisualization,
        "image_boxes": DrawingBoxVisualization,
        "batch_boxes": BatchedDrawingBoxVisualization,
        "prep": BatchedDetectionDrawingPrep,
    }
    calls: Dict[str, int] = {name: 0 for name in [*counted, "transfers"]}
    for name, block_class in counted.items():

        def counting(self, *, _original=block_class.run, _name=name, **kwargs):
            calls[_name] += 1
            return _original(self, **kwargs)

        monkeypatch.setattr(block_class, "run", counting)

    def counting_transfer(tensors, _original=drawing_blocks._to_host):
        calls["transfers"] += 1
        return _original(tensors)

    monkeypatch.setattr(drawing_blocks, "_to_host", counting_transfer)
    frames = PARITY_10.marker_frames(4, sizes_hw=FRAME_SIZES)
    # Batch painters still paint every image with the per-image painter.
    expected = {
        "per-painter": {"per_painter_boxes": 4},
        "shared-prep": {"image_boxes": 4, "prep": 1, "transfers": 1},
        "batch-painters": {
            "batch_boxes": 1,
            "image_boxes": 4,
            "prep": 1,
            "transfers": 1,
        },
    }

    for drawing, expected_calls in expected.items():
        backend = drawing_backend.build_for_model(
            "v2_serial",
            model=ScriptedDetectionModel(),
            model_id="scripted",
            confidence=0.4,
            drawing=drawing,
        )
        before = dict(calls)
        rows = backend.process_batch(frames, image_ids=["a", "b", "c", "d"])

        made = {name: calls[name] - before[name] for name in calls}
        assert made == {name: expected_calls.get(name, 0) for name in calls}
        if drawing != "per-painter":
            assert len({row["prep_ms"] for row in rows}) == 1  # one batch time


@pytest.mark.parametrize(
    "batch_size, expected",
    [
        (8, {"per-painter": 17, "shared-prep": 18, "batch-painters": 4}),
        (1, {"per-painter": 3, "shared-prep": 4, "batch-painters": 4}),
    ],
)
def test_engine_calls_per_run_from_the_plan(
    batch_size: int, expected: Dict[str, int]
) -> None:
    import run_structural

    totals = {
        drawing: run_structural._engine_calls(
            run_structural.Case(drawing, batch_size, 0), stages="full"
        )["total"]
        for drawing in drawing_backend.DRAWINGS
    }

    assert totals == expected


def test_shared_drawings_need_v2_and_a_painter_stage() -> None:
    with pytest.raises(ValueError, match="painter|boxes"):
        drawing_backend.v2_workflow("detector", drawing="shared-prep")
    with pytest.raises(ValueError, match="V2 mode"):
        drawing_backend.build_for_model(
            "v1_tensor",
            model=ScriptedDetectionModel(),
            model_id="scripted",
            confidence=0.4,
            drawing="batch-painters",
        )
    with pytest.raises(ValueError, match="drawing must be"):
        drawing_backend.v2_workflow("full", drawing="fused")


def test_per_painter_is_the_unchanged_10_workflow() -> None:
    import batched_backend

    for stages in drawing_backend.STAGES:
        definition = drawing_backend.v2_workflow(stages, drawing="per-painter")
        assert definition == batched_backend.v2_workflow(stages)


def test_run_live_swaps_the_10_backend_factory_only_during_the_run() -> None:
    import run_live
    from click.testing import CliRunner

    run_batched = structural_imports.load_10_module("run_batched")
    original = run_batched.batched_backend

    with run_live._drawing_backends(run_batched, drawing="batch-painters"):
        backend = run_batched.batched_backend.build_for_model(
            "v2_serial",
            model=ScriptedDetectionModel(),
            model_id="scripted",
            confidence=0.4,
        )
        backend.close()

    assert backend.facts["drawing"] == "batch-painters"
    assert run_batched.batched_backend is original
    refused = CliRunner().invoke(
        run_live.main, ["--drawing", "shared-prep", "--mode", "v1_tensor"]
    )
    assert refused.exit_code == 2 and "V2 --mode" in refused.output


def test_inspect_lifetimes_reports_and_restores_gc() -> None:
    # No assertion on the alive count: the engine fix is pending.
    import gc

    import inspect_lifetimes

    enabled_before = gc.isenabled()

    report = inspect_lifetimes.inspect_lifetimes(runs=4)

    assert gc.isenabled() == enabled_before
    assert 0 <= report["payloads_alive_after_results_dropped"] <= 4
    assert report["payloads_alive_after_cyclic_collection"] == 0
    assert set(report["engine_sources"]) == {
        "roboflow_workflows.execution_engine.v2.execution.entries",
        "roboflow_workflows.execution_engine.v2.execution.outputs",
    }


def test_empty_drawing_preparation_returns_empty_batch():
    assert BatchedDetectionDrawingPrep().run(predictions=Batch([])) == []


@pytest.mark.parametrize(
    "painter", [BatchedDrawingBoxVisualization, BatchedDrawingLabelVisualization]
)
def test_batch_painters_reject_mismatch_before_painting(painter):
    block = painter()
    calls = []
    block._painter.run = lambda **kwargs: calls.append(kwargs)
    with pytest.raises(ValueError, match="batches must align"):
        block.run(image=Batch([object()]), drawing=Batch([]))
    assert not calls
