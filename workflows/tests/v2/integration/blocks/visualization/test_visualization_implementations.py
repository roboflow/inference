"""One public box/label contract, selectable rendering and unchanged ownership."""

import pytest
import torch
from pydantic import Field
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.blocks.visualization import (
    BoundingBoxVisualizationBlock,
    LabelVisualizationBlock,
)
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
from roboflow_workflows.v2.engine.declaration import spec_of

from tests.v2.unit.blocks.visualization.annotations.support import (
    detections,
    image_of,
    scene,
)


class FixedDetections(Block):
    type = "test/visualization_detections@v1"
    outputs = {"predictions": Output(OBJECT_DETECTION_PREDICTION_KIND, source="image")}

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image for synthetic detections.")

    def run(self, *, image):
        result = {"predictions": detections(image)}

        return result


def _definition(execution=None):
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowBatchInput", "name": "images", "kind": ["image"]}],
        "steps": [
            {"type": FixedDetections.type, "name": "detect", "image": "$inputs.images"},
            {
                "type": BoundingBoxVisualizationBlock.type,
                "name": "boxes",
                "image": "$inputs.images",
                "predictions": "$steps.detect.predictions",
            },
            {
                "type": LabelVisualizationBlock.type,
                "name": "labels",
                "image": "$steps.boxes.image",
                "predictions": "$steps.detect.predictions",
                "text_size_mode": "Manual",
                "text_scale": 0.4,
            },
        ],
        "outputs": [
            {"type": "JsonField", "name": "boxes", "selector": "$steps.boxes.image"},
            {"type": "JsonField", "name": "labels", "selector": "$steps.labels.image"},
        ],
    }
    if execution is not None:
        definition["execution"] = execution

    return definition


@pytest.mark.parametrize(
    "execution,deployment,expected",
    [
        (None, None, ("accurate", "accurate")),
        ({"quality": "fast"}, None, ("fast", "fast")),
        (None, "fast", ("fast", "fast")),
        (
            {"quality": "fast", "step_quality": {"$steps.boxes": "accurate"}},
            None,
            ("accurate", "fast"),
        ),
    ],
)
def test_quality_selection_runs_two_images_without_mutating_observable_outputs(
    execution, deployment, expected
):
    catalogue = Catalogue(
        [FixedDetections, BoundingBoxVisualizationBlock, LabelVisualizationBlock]
    )
    plan = compile_workflow(
        _definition(execution),
        catalogue=catalogue,
        options=CompileOptions(quality=deployment),
    )
    assert (
        tuple(plan.step((name,)).implementation.name for name in ("boxes", "labels"))
        == expected
    )
    images = [image_of(scene(seed=i), image_id=f"source-{i}") for i in range(2)]
    originals = [image.tensor_image.clone() for image in images]

    with plan.create_session() as session:
        results = session.run({"images": images}).rows()

    assert len(results) == len(images)
    for image, original, result in zip(images, originals, results):
        assert torch.equal(image.tensor_image, original)
        assert result["boxes"].image_id == result["labels"].image_id == image.image_id
        assert result["boxes"].root == result["labels"].root == image.root
        assert (
            result["boxes"].tensor_image.data_ptr()
            != result["labels"].tensor_image.data_ptr()
        )
        assert result["boxes"].tensor_image.data_ptr() != image.tensor_image.data_ptr()
        assert not torch.equal(
            result["boxes"].tensor_image, result["labels"].tensor_image
        )


def test_catalogue_exposes_only_canonical_visualization_blocks():
    catalogue = create_catalogue()
    for block in (BoundingBoxVisualizationBlock, LabelVisualizationBlock):
        assert catalogue.block_types.count(block.type) == 1
        assert {item.name for item in spec_of(block).implementations} == {
            "fast",
            "accurate",
        }

    for family in (
        "simple_box_visualization",
        "simple_label_visualization",
        "simple_label_visualization_in_place",
        "detection_drawing",
    ):
        assert catalogue.find(f"roboflow_core/{family}@v1") is None
