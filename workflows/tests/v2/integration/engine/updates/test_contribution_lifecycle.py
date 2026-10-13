"""A host-supplied implementation of the real ``v2/box_visualization@v1`` block.

The contributed implementation paints each box's four border bands with
NumPy slicing, one box at a time on the host: an independent route to the band
geometry the public tensor kernel documents (``thickness``-wide bands centred
on the box edge, clipped to the frame, later boxes painted over earlier ones).
The public parity criterion for deterministic drawing is pixel identity: the
same workflow definition must return identical images with and without the
contribution. Calls the implementation does not support are checked before any
work, delegated to the public block, and counted.

It runs on CPU and returns complete images, which meets the readiness rule of
contributions without any stream handling.
"""

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks import visualization
from roboflow_workflows.v2.blocks.visualization.blocks import (
    BoxVisualizationBlock,
)
from roboflow_workflows.v2.blocks.visualization.drawing import (
    BOX_THICKNESS,
    PALETTE,
    DetectionDrawing,
    prepare_drawings,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine import (
    Block,
    BlockParams,
    Catalogue,
    CompiledWorkflow,
    CompileOptions,
    Implementation,
    ImplementationContribution,
    Output,
    Ref,
    Target,
    compile_workflow,
    contract_digest,
)
from roboflow_workflows.v2.engine.errors import (
    GraphUpdateError,
    StepExecutionError,
)
from roboflow_workflows.v2.engine.updates import compare_plans

from inference_models.models.base.object_detection import Detections

CLASS_NAMES = {0: "person", 1: "car", 2: "dog", 5: "bus"}
HOST_DRAWING = Target(frozenset({"cpu", "host-drawing"}))


class FixedDetector(Block):
    """Emits the caller's ``Detections`` for each image id, as one batch."""

    type = "test/fixed_detector@v1"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="Images to 'detect' on."
        )

    def __init__(self, *, detections_by_image: Dict[str, Detections]):
        self._by_image = detections_by_image

    def run(self, *, image: Batch) -> List[Dict[str, Any]]:
        results = [{"predictions": self._by_image[one.image_id]} for one in image]

        return results


class HostBoxes(Implementation):
    """Box borders painted per box with NumPy on CPU images; same pixels as the block."""

    name = "host-boxes"
    requires = ("cpu", "host-drawing")

    def __init__(self) -> None:
        self.reference = BoxVisualizationBlock()
        self.delegated = 0

    def run(self, *, image: Batch, predictions: Batch) -> List[Dict[str, Any]]:
        images, members = list(image), list(predictions)
        drawings = self._supported_drawings(images, members)
        if drawings is None:
            self.delegated += 1
            delegated = self.reference.run(image=image, predictions=predictions)
            return delegated

        results = [
            {"image": member.with_pixels(self._paint(member.tensor_image, drawing))}
            for member, drawing in zip(images, drawings)
        ]

        return results

    def _supported_drawings(
        self, images: List[ImageData], members: List[Any]
    ) -> Optional[List[DetectionDrawing]]:
        # Preflight only: nothing is painted before every check passed.
        if len(images) != len(members):
            return None
        if any(
            member.channels != 3 or member.tensor_image.device.type != "cpu"
            for member in images
        ):
            return None

        native = [one for one in members if not isinstance(one, DetectionDrawing)]
        prepared = iter(prepare_drawings(native))
        drawings = [
            one if isinstance(one, DetectionDrawing) else next(prepared)
            for one in members
        ]
        if any((np.asarray(one.class_id) < 0).any() for one in drawings):
            return None

        return drawings

    def _paint(self, pixels: torch.Tensor, drawing: DetectionDrawing) -> torch.Tensor:
        scene = pixels.permute(1, 2, 0).numpy().copy()
        height, width = scene.shape[:2]
        thickness, outer = BOX_THICKNESS, BOX_THICKNESS // 2
        corners = np.asarray(drawing.xyxy).reshape(-1, 4).astype(np.int64)
        for (a, b, c, d), class_id in zip(corners, drawing.class_id):
            left, right = min(a, c), max(a, c)
            top, bottom = min(b, d), max(b, d)
            x1, x2, y1, y2 = left - outer, right + outer, top - outer, bottom + outer
            color = PALETTE.by_idx(int(class_id)).as_rgb()
            bands = (
                (y1, y1 + thickness - 1, x1, x2),
                (y2 - thickness + 1, y2, x1, x2),
                (top, bottom, x1, x1 + thickness - 1),
                (top, bottom, x2 - thickness + 1, x2),
            )
            for row_from, row_to, column_from, column_to in bands:
                # Inclusive band corners, clipped to the frame; empty off-frame.
                row_from, row_to = max(row_from, 0), min(row_to, height - 1)
                column_from, column_to = max(column_from, 0), min(column_to, width - 1)
                if row_from > row_to or column_from > column_to:
                    continue
                scene[row_from : row_to + 1, column_from : column_to + 1] = color

        painted = torch.from_numpy(scene).permute(2, 0, 1).contiguous()

        return painted


def host_boxes(version: str = "1.0.0") -> ImplementationContribution:
    contribution = ImplementationContribution(
        block_type=BoxVisualizationBlock.type,
        contract=contract_digest(BoxVisualizationBlock),
        implementations=(HostBoxes,),
        provider="example-provider",
        version=version,
    )

    return contribution


def public() -> Catalogue:
    catalogue = Catalogue(
        (FixedDetector, *visualization.BLOCKS), kinds=list(visualization.KINDS)
    )

    return catalogue


def detections(xyxy: Sequence[Sequence[float]], class_id: Sequence[int]) -> Detections:
    found = Detections(
        xyxy=torch.tensor(xyxy, dtype=torch.float32).reshape(-1, 4),
        class_id=torch.tensor(class_id, dtype=torch.int64),
        confidence=torch.full((len(class_id),), 0.5, dtype=torch.float32),
        image_metadata={"class_names": dict(CLASS_NAMES)},
    )

    return found


def scene() -> Dict[str, Detections]:
    # Overlapping boxes, boxes crossing the top-left and bottom-right corners,
    # fractional coordinates, and an image without detections.
    by_image = {
        "a": detections(
            [
                [10, 30, 80, 90],
                [50.6, 40.4, 150.2, 115.9],
                [-5, -5, 30, 20],
                [140, 100, 170, 130],
            ],
            [0, 1, 2, 5],
        ),
        "b": detections([[0, 0, 159, 119], [20, 20, 21, 21]], [5, 0]),
        "empty": detections([], []),
    }

    return by_image


def images() -> List[ImageData]:
    loaded = []
    for seed, image_id in enumerate(("a", "b", "empty")):
        generator = torch.Generator().manual_seed(seed)
        pixels = torch.randint(
            0, 256, (3, 120, 160), generator=generator, dtype=torch.uint8
        )
        loaded.append(ImageData.from_tensor(pixels, image_id=image_id))

    return loaded


def definition(*, prepared: bool = False) -> dict:
    painter = {
        "type": BoxVisualizationBlock.type,
        "name": "boxes",
        "image": "$inputs.image",
        "predictions": "$steps.detector.predictions",
    }
    steps = [
        {"type": "test/fixed_detector@v1", "name": "detector", "image": "$inputs.image"}
    ]
    if prepared:
        steps.append(
            {
                "type": "v2/detection_drawing@v1",
                "name": "drawing",
                "predictions": "$steps.detector.predictions",
            }
        )
        painter["predictions"] = "$steps.drawing.drawing"
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "image", "kind": ["image"]}],
        "steps": steps + [painter],
        "outputs": [
            {"type": "JsonField", "name": "annotated", "selector": "$steps.boxes.image"}
        ],
    }

    return workflow


def compiled(
    catalogue: Catalogue, target: Target = HOST_DRAWING, **settings: Any
) -> CompiledWorkflow:
    plan = compile_workflow(
        definition(**settings),
        catalogue=catalogue,
        options=CompileOptions(target=target),
    )

    return plan


def annotate(plan: CompiledWorkflow) -> List[ImageData]:
    session = plan.create_session({"detections_by_image": scene()})
    annotated = [row["annotated"] for row in session.run({"image": images()}).rows()]

    return annotated


@pytest.mark.parametrize("prepared", [False, True], ids=["detections", "drawings"])
def test_contributed_painter_draws_the_public_blocks_pixels(prepared: bool) -> None:
    public_plan = compiled(public(), prepared=prepared)
    plan = compiled(public().with_contributions([host_boxes()]), prepared=prepared)

    expected = annotate(public_plan)
    session = plan.create_session({"detections_by_image": scene()})
    actual = [row["annotated"] for row in session.run({"image": images()}).rows()]

    assert public_plan.step(("boxes",)).selected.name == "default"
    assert plan.step(("boxes",)).selected.name == "host-boxes"
    assert session.instances[("boxes",)].delegated == 0
    for got, wanted, source in zip(actual, expected, images()):
        assert torch.equal(got.tensor_image, wanted.tensor_image)
        assert got.image_id == wanted.image_id == source.image_id
    assert not torch.equal(actual[0].tensor_image, images()[0].tensor_image)
    assert torch.equal(actual[2].tensor_image, images()[2].tensor_image)


def test_target_without_the_capability_keeps_the_public_painter() -> None:
    plan = compiled(public().with_contributions([host_boxes()]), Target.cpu())

    assert plan.step(("boxes",)).selected.name == "default"
    assert plan.step(("boxes",)).implementation.considered == (
        ("host-boxes", ("host-drawing",)),
        ("default", ()),
    )
    for got, wanted in zip(annotate(plan), annotate(compiled(public()))):
        assert torch.equal(got.tensor_image, wanted.tensor_image)


def test_unsupported_calls_are_delegated_before_any_work_and_counted() -> None:
    # A negative class id has no palette color: the public block refuses it.
    unsupported = {
        image_id: Detections(
            xyxy=torch.tensor([[10.0, 10.0, 40.0, 40.0]]),
            class_id=torch.tensor([-1]),
            confidence=torch.tensor([0.5]),
            image_metadata={"class_names": {-1: "unknown"}},
        )
        for image_id in ("a", "b", "empty")
    }
    public_session = compiled(public()).create_session(
        {"detections_by_image": unsupported}
    )
    session = compiled(public().with_contributions([host_boxes()])).create_session(
        {"detections_by_image": unsupported}
    )

    with pytest.raises(StepExecutionError) as public_error:
        public_session.run({"image": images()})
    with pytest.raises(StepExecutionError) as contributed_error:
        session.run({"image": images()})

    assert session.instances[("boxes",)].delegated == 1
    assert isinstance(contributed_error.value.__cause__, ValueError)
    assert str(contributed_error.value.__cause__) == str(public_error.value.__cause__)


def test_public_plan_description_has_no_origin_and_the_contributed_one_does() -> None:
    public_described = compiled(public()).describe()
    described = compiled(public().with_contributions([host_boxes()])).describe()

    painter = [step for step in described["steps"] if step["path"] == "$steps.boxes"]
    public_painter = [
        step for step in public_described["steps"] if step["path"] == "$steps.boxes"
    ]
    assert all(
        "origin" not in step["implementation"] for step in public_described["steps"]
    )
    assert painter[0]["implementation"]["name"] == "host-boxes"
    assert painter[0]["implementation"]["origin"] == {
        "provider": "example-provider",
        "version": "1.0.0",
        "contract": contract_digest(BoxVisualizationBlock),
    }
    assert public_painter[0]["implementation"]["name"] == "default"


def test_provider_version_update_resets_and_reactivation_preserves() -> None:
    old = compiled(public().with_contributions([host_boxes("1.0.0")]))
    same = compiled(public().with_contributions([host_boxes("1.0.0")]))
    new = compiled(public().with_contributions([host_boxes("1.1.0")]))
    session = old.create_session({"detections_by_image": scene()})
    first = session.instances[("boxes",)]
    before = [row["annotated"] for row in session.run({"image": images()}).rows()]

    assert compare_plans(old, same).kind == "preserve"
    preserved = session.update(same)
    assert not preserved.reset and session.instances[("boxes",)] is first

    with pytest.raises(GraphUpdateError):
        session.update(new)
    receipt = session.update(new, reset=True)
    after = [row["annotated"] for row in session.run({"image": images()}).rows()]

    assert receipt.reset
    assert session.instances[("boxes",)] is not first
    for one, other in zip(before, after):
        assert torch.equal(one.tensor_image, other.tensor_image)
