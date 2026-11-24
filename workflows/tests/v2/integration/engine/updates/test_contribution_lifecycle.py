"""Canonical per-image painter contributions preserve the generic lifecycle.

A host implementation draws square borders independently with OpenCV and
falls back to the accurate implementation for unsupported styles/classes.
Selection, origin metadata, version resets and reactivation remain covered.
"""

from typing import Any, Dict, List, Sequence

import cv2
import pytest
import supervision as sv
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks import visualization
from roboflow_workflows.v2.blocks.visualization.annotations.bounding_box_visualization.v1 import (
    AccurateBoundingBoxes,
)
from roboflow_workflows.v2.blocks.visualization.annotations.bounding_box_visualization.v1 import (
    BoundingBoxVisualizationBlock as BoxVisualizationBlock,
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
    """Independent CPU square-box renderer with reference fallback."""

    name = "host-boxes"
    requires = ("cpu", "host-drawing")
    quality = ("accurate",)

    def __init__(self) -> None:
        self.reference = AccurateBoundingBoxes()
        self.delegated = 0

    def run(
        self,
        *,
        image,
        predictions,
        color_palette,
        palette_size,
        custom_colors,
        color_axis,
        thickness,
        roundness
    ):
        arguments = dict(
            image=image,
            predictions=predictions,
            color_palette=color_palette,
            palette_size=palette_size,
            custom_colors=custom_colors,
            color_axis=color_axis,
            thickness=thickness,
            roundness=roundness,
        )
        if (
            image.channels != 3
            or image.tensor_image.device.type != "cpu"
            or color_palette != "DEFAULT"
            or color_axis != "CLASS"
            or roundness != 0
            or (predictions.class_id < 0).any()
        ):
            self.delegated += 1
            return self.reference.run(**arguments)

        scene = image.tensor_image.permute(1, 2, 0).numpy().copy()
        for corners, class_id in zip(
            predictions.xyxy.int().tolist(), predictions.class_id.tolist()
        ):
            x1, y1, x2, y2 = corners
            cv2.rectangle(
                scene,
                (x1, y1),
                (x2, y2),
                sv.ColorPalette.DEFAULT.by_idx(class_id).as_rgb(),
                thickness,
            )
        painted = torch.from_numpy(scene).permute(2, 0, 1).contiguous()
        return {"image": image.with_pixels(painted)}


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
    catalogue = Catalogue((FixedDetector, *visualization.BLOCKS))

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


def definition() -> dict:
    painter = {
        "type": BoxVisualizationBlock.type,
        "name": "boxes",
        "image": "$inputs.image",
        "predictions": "$steps.detector.predictions",
    }
    steps = [
        {"type": "test/fixed_detector@v1", "name": "detector", "image": "$inputs.image"}
    ]
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


def test_contributed_painter_draws_the_public_blocks_pixels() -> None:
    public_plan = compiled(public())
    plan = compiled(public().with_contributions([host_boxes()]))

    expected = annotate(public_plan)
    session = plan.create_session({"detections_by_image": scene()})
    actual = [row["annotated"] for row in session.run({"image": images()}).rows()]

    assert public_plan.step(("boxes",)).selected.name == "accurate"
    assert plan.step(("boxes",)).selected.name == "host-boxes"
    assert session.instances[("boxes",)].delegated == 0
    for got, wanted, source in zip(actual, expected, images()):
        assert torch.equal(got.tensor_image, wanted.tensor_image)
        assert got.image_id == wanted.image_id == source.image_id
    assert not torch.equal(actual[0].tensor_image, images()[0].tensor_image)
    assert torch.equal(actual[2].tensor_image, images()[2].tensor_image)


def test_target_without_the_capability_keeps_the_public_painter() -> None:
    plan = compiled(public().with_contributions([host_boxes()]), Target.cpu())

    assert plan.step(("boxes",)).selected.name == "accurate"
    assert plan.step(("boxes",)).implementation.considered == (
        ("host-boxes", ("host-drawing",)),
        ("accurate", ()),
        ("fast", ()),
    )
    for got, wanted in zip(annotate(plan), annotate(compiled(public()))):
        assert torch.equal(got.tensor_image, wanted.tensor_image)


def test_unsupported_calls_are_delegated_before_any_work_and_counted() -> None:
    # Negative ids require canonical grey coloring, outside the host fast path.
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

    expected = public_session.run({"image": images()}).rows()
    actual = session.run({"image": images()}).rows()
    assert session.instances[("boxes",)].delegated == 3
    for got, wanted in zip(actual, expected):
        assert torch.equal(
            got["annotated"].tensor_image, wanted["annotated"].tensor_image
        )


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
    assert public_painter[0]["implementation"]["name"] == "accurate"


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
