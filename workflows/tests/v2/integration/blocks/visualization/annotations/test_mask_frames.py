"""Compiled: masks segmented on crops are drawn at their place on the parent.

The real instance segmentation and dynamic crop blocks feed the mask readers;
each crop is segmented again and its masks are drawn on the parent image::

    image ─▶ seg ─▶ dynamic crop ─▶ inner seg ─▶ mask / polygon / label
      └──────────────────────────────────────────▶ (drawn on the parent)

Street 112x72 with three objects whose boxes do not overlap, so each crop
holds one object. A crop's mask, pasted at the crop origin, is the painted
shape itself.

Oracle: the latest V1 block given that one row in the parent frame: the
shape's box and its painted mask (``scene.shape_mask``). A synthetic compact
inner model (stride-4 logit grid, ``mask_quality`` accurate) is compared with
V1 given its grid materialized in the crop and pasted at the crop origin.
"""

from typing import Any, Dict

import numpy as np
import pytest
import supervision as sv
import torch
from roboflow_workflows.core_steps.visualizations.label.v2 import (
    LabelVisualizationBlockV2,
)
from roboflow_workflows.core_steps.visualizations.mask.v1 import (
    MaskVisualizationBlockV1,
)
from roboflow_workflows.core_steps.visualizations.polygon.v2 import (
    PolygonVisualizationBlockV2,
)
from roboflow_workflows.execution_engine.entities.base import (
    ImageParentMetadata,
    WorkflowImageData,
)
from roboflow_workflows.v2.blocks.models.cache import ModelCache
from roboflow_workflows.v2.blocks.visualization.annotations import (
    LabelVisualizationBlock,
    MaskVisualizationBlock,
    PolygonVisualizationBlock,
)
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.plan import CompileOptions

from inference_models.models.base.instance_segmentation import InstanceDetections
from tests.v2.integration.blocks.visualization.annotations.scene import (
    CLASS_NAMES,
    OBJECT_COLORS,
    STREET,
    STREET_SIZE,
    SceneLoader,
    annotation_catalogue,
    compact_view,
    pixels_of,
    shape_mask,
    street_image,
)

COLORS = {
    "color_palette": "DEFAULT",
    "palette_size": 10,
    "custom_colors": [],
    "color_axis": "CLASS",
}
# V2 block -> (latest V1 block, parameters)
DRAWINGS = {
    MaskVisualizationBlock: (MaskVisualizationBlockV1, {"opacity": 0.5}),
    PolygonVisualizationBlock: (PolygonVisualizationBlockV2, {"thickness": 2}),
    LabelVisualizationBlock: (
        LabelVisualizationBlockV2,
        {
            "text": "Area",
            "text_position": "CENTER_OF_MASS",
            "text_color": "WHITE",
            "text_size_mode": "Manual",
            "text_scale": 0.4,
            "text_thickness": 1,
            "text_padding": 3,
            "border_radius": 2,
        },
    ),
}
INNER_MODELS = {
    "dense": "v2/instance_segmentation@v1",
    "compact": "test/synthetic_compact_segmentation@v1",
}


def definition(inner_type: str) -> Dict[str, Any]:
    steps = [
        {
            "type": "v2/instance_segmentation@v1",
            "name": "seg",
            "image": "$inputs.image",
            "model_id": "shapes-seg",
        },
        {
            "type": "v2/dynamic_crop@v1",
            "name": "crop",
            "image": "$inputs.image",
            "predictions": "$steps.seg.predictions",
        },
        {
            "type": inner_type,
            "name": "inner",
            "image": "$steps.crop.crops",
            "model_id": "shapes-seg",
        },
    ]
    outputs = [
        {"type": "JsonField", "name": "crops", "selector": "$steps.crop.crops"},
        {"type": "JsonField", "name": "inner", "selector": "$steps.inner.predictions"},
    ]
    for index, (block_class, (_, parameters)) in enumerate(DRAWINGS.items()):
        name = f"draw_{index}"
        steps.append(
            {
                "type": block_class.type,
                "name": name,
                "image": "$inputs.image",
                "predictions": "$steps.inner.predictions",
                "mask_quality": "accurate",
                **parameters,
                **COLORS,
            }
        )
        outputs.append(
            {"type": "JsonField", "name": name, "selector": f"$steps.{name}.image"}
        )
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}],
        "steps": steps,
        "outputs": outputs,
    }

    return workflow


def crop_masks(inner: str, shape_index: int) -> np.ndarray:
    """``(1, h, w)`` bool mask of one shape as the inner model sees its crop."""
    shape = STREET[shape_index]
    x0, y0, x1, y1 = shape.xyxy
    dense = shape_mask(shape, STREET_SIZE)[None, y0:y1, x0:x1]
    if inner == "dense":
        return dense

    rows = InstanceDetections(
        xyxy=torch.tensor([[0.0, 0.0, x1 - x0, y1 - y0]]),
        class_id=torch.tensor([0]),
        confidence=torch.tensor([1.0]),
        mask=torch.from_numpy(dense),
    )
    materialized = compact_view(rows).full_res().mask.cpu().numpy()

    return materialized


def v1_drawing(
    v1_class: type,
    parent: np.ndarray,
    shape_index: int,
    masks: np.ndarray,
    **parameters: Any,
) -> np.ndarray:
    """The latest V1 block drawing one shape's row on the parent pixels."""
    shape = STREET[shape_index]
    class_id, confidence = OBJECT_COLORS[shape.color]
    x0, y0, x1, y1 = shape.xyxy
    parent_masks = np.zeros((1, *STREET_SIZE), dtype=bool)
    parent_masks[:, y0:y1, x0:x1] = masks
    predictions = sv.Detections(
        xyxy=np.array([shape.xyxy], dtype=np.float32),
        class_id=np.array([class_id]),
        confidence=np.array([confidence], dtype=np.float32),
        mask=parent_masks,
        data={"class_name": np.array([CLASS_NAMES[class_id]])},
    )
    image = WorkflowImageData(
        parent_metadata=ImageParentMetadata(parent_id="street"),
        numpy_image=np.ascontiguousarray(parent[:, :, ::-1]),
    )
    result = v1_class().run(
        image=image, predictions=predictions, copy_image=True, **parameters
    )
    drawing = result["image"].numpy_image[:, :, ::-1]

    return drawing


@pytest.mark.parametrize("inner", list(INNER_MODELS))
def test_masks_segmented_on_crops_are_drawn_at_their_place_on_the_parent(
    inner: str,
) -> None:
    # given
    image = street_image()
    parent = pixels_of(image)
    plan = compile_workflow(
        definition(INNER_MODELS[inner]),
        catalogue=annotation_catalogue(),
        options=CompileOptions(block_execution="run"),
    )
    session = plan.create_session({"models": ModelCache(SceneLoader())})

    # when
    (row,) = session.run({"image": [image]}).rows()

    # then: one crop per object, each segmented again as one row; workflow
    # outputs report it in the input image, at the shape's box
    assert len(row["crops"]) == len(STREET)
    for crop, rows, shape in zip(row["crops"], row["inner"], STREET):
        x0, y0, x1, y1 = shape.xyxy
        np.testing.assert_array_equal(pixels_of(crop), parent[y0:y1, x0:x1])
        assert rows.xyxy.tolist() == [list(map(float, shape.xyxy))]

    # each drawing on the parent equals V1 given that row in the parent frame
    for index, (_, (v1_class, parameters)) in enumerate(DRAWINGS.items()):
        drawings = row[f"draw_{index}"]
        assert len(drawings) == len(STREET)
        for shape_index, drawn in enumerate(drawings):
            assert drawn.image_id == image.image_id
            expected = v1_drawing(
                v1_class,
                parent,
                shape_index,
                crop_masks(inner, shape_index),
                **parameters,
                **COLORS,
            )
            differing = (pixels_of(drawn) != expected).any(axis=-1)
            assert not differing.any(), (
                f"{v1_class.__name__} on crop {shape_index}: "
                f"{int(differing.sum())} pixels differ from V1"
            )

    # the input image is unchanged
    np.testing.assert_array_equal(pixels_of(image), parent)
