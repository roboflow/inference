"""Compiled workflows select and materialize grid, dense and RLE masks alike."""

from typing import List

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from pydantic import Field
from roboflow_workflows.v2.blocks import create_catalogue
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.kinds import STRING_KIND
from roboflow_workflows.v2.data.masks import (
    INSTANCE_SEGMENTATION_VIEW_KIND,
    MaskGridGeometry,
    SegmentationView,
    mask_provider_of,
)
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks

ROOT_HW = (20, 30)
CROP_XYXY = (5, 4, 16, 13)  # a 9 x 11 crop at (5, 4)
HEIGHT, WIDTH = 9, 11


class _SelectMasks(Block):
    type = "mask_test/select_masks"
    outputs = {
        "masks": Output(INSTANCE_SEGMENTATION_PREDICTION_KIND, source="predictions"),
        "fidelity": Output(STRING_KIND, source="predictions"),
    }

    class Params(BlockParams):
        predictions: Ref(
            INSTANCE_SEGMENTATION_VIEW_KIND,
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
            RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
        ) = Field(description="Instance masks in any supported representation.")
        indices: List[int] = Field(description="Rows to keep, in order.")
        fidelity: str = Field(description="reference or approximate.")

    def run(self, *, predictions, indices, fidelity):
        provider = mask_provider_of(predictions).select(indices=indices)
        masks = provider.binary(fidelity=fidelity)

        return {"masks": masks.predictions, "fidelity": masks.fidelity}


def _crop() -> ImageData:
    root = ImageData.from_tensor(
        torch.zeros((3, *ROOT_HW), dtype=torch.uint8), image_id="root"
    )
    crop = root.crop(CROP_XYXY, image_id="crop")

    return crop


def _masks() -> torch.Tensor:
    masks = torch.zeros((3, HEIGHT, WIDTH), dtype=torch.bool)
    masks[0, 1:7, 1:7] = True
    masks[0, 3:5, 3:5] = False  # a hole
    masks[1, 0:2, 0:3] = True
    masks[1, 6:9, 8:11] = True  # a second, separate region
    masks[2, 8, 10] = True

    return masks


def _fields(metadata) -> dict:
    fields = {
        "xyxy": torch.tensor(
            [[1, 1, 7, 7], [0, 0, 11, 9], [10, 8, 11, 9]], dtype=torch.float32
        ),
        "class_id": torch.tensor([4, 5, 6]),
        "confidence": torch.tensor([0.9, 0.8, 0.7]),
        "image_metadata": metadata,
        "bboxes_metadata": [{"row": 0}, {"row": 1}, {"row": 2}],
    }

    return fields


def _representations(metadata) -> dict:
    """The same three analytic masks as grid scores, dense masks and COCO RLE."""
    masks = _masks()
    # Logits +1 / -1 on a grid the size of the image: the bilinear resize to
    # the same size is the identity, so the reference equals ``masks``.
    view = SegmentationView(
        detections=Detections(**_fields(metadata)),
        scores=masks.float() * 2 - 1,
        score_type="logits",
        threshold=0.0,
        geometry=MaskGridGeometry.identity((HEIGHT, WIDTH)),
    )
    encoded = mask_utils.encode(
        np.asfortranarray(masks.numpy().transpose(1, 2, 0).astype(np.uint8))
    )
    representations = {
        "instance_segmentation_view": view,
        "instance_segmentation_prediction": InstanceDetections(
            **_fields(metadata), mask=masks
        ),
        "rle_instance_segmentation_prediction": InstanceDetections(
            **_fields(metadata),
            mask=InstancesRLEMasks.from_coco_rle_masks((HEIGHT, WIDTH), encoded),
        ),
    }

    return representations


def _definition(kind: str, *, indices, fidelity: str) -> dict:
    definition = {
        "version": "2.0",
        "inputs": [
            {"type": "WorkflowParameter", "name": "predictions", "kind": [kind]}
        ],
        "steps": [
            {
                "type": "mask_test/select_masks",
                "name": "select",
                "predictions": "$inputs.predictions",
                "indices": indices,
                "fidelity": fidelity,
            }
        ],
        "outputs": [
            {
                "type": "JsonField",
                "name": "own",
                "selector": "$steps.select.masks",
                "coordinates_system": "own",
            },
            {
                "type": "JsonField",
                "name": "root",
                "selector": "$steps.select.masks",
                "coordinates_system": "root",
            },
            {
                "type": "JsonField",
                "name": "fidelity",
                "selector": "$steps.select.fidelity",
            },
        ],
    }

    return definition


def _run(kind: str, payload, *, indices, fidelity="reference") -> dict:
    plan = compile_workflow(
        _definition(kind, indices=indices, fidelity=fidelity),
        catalogue=create_catalogue().with_blocks([_SelectMasks]),
    )
    (row,) = plan.create_session().run({"predictions": payload}).rows()

    return row


KINDS = [
    "instance_segmentation_view",
    "instance_segmentation_prediction",
    "rle_instance_segmentation_prediction",
]


@pytest.mark.parametrize("kind", KINDS)
def test_selected_masks_are_identical_across_representations(kind) -> None:
    metadata = _crop().prediction_metadata()
    payload = _representations(metadata)[kind]

    row = _run(kind, payload, indices=[2, 0, 2])

    own = row["own"]
    assert row["fidelity"] == "reference"
    assert torch.equal(own.mask, _masks()[[2, 0, 2]])
    assert own.class_id.tolist() == [6, 4, 6]
    assert own.bboxes_metadata == [{"row": 2}, {"row": 0}, {"row": 2}]
    assert not own.mask[1, 3:5, 3:5].any()


@pytest.mark.parametrize("kind", KINDS)
def test_root_conversion_is_the_separate_explicit_output_step(kind) -> None:
    metadata = _crop().prediction_metadata()
    payload = _representations(metadata)[kind]

    row = _run(kind, payload, indices=[1])

    own, root = row["own"], row["root"]
    x0, y0 = CROP_XYXY[:2]
    assert tuple(own.mask.shape) == (1, HEIGHT, WIDTH)
    assert tuple(root.mask.shape) == (1, *ROOT_HW)
    assert torch.equal(root.mask[0, y0 : y0 + HEIGHT, x0 : x0 + WIDTH], own.mask[0])
    assert int(root.mask.sum()) == int(own.mask.sum())
    assert root.xyxy.tolist() == [[x0, y0, x0 + WIDTH, y0 + HEIGHT]]


@pytest.mark.parametrize("kind", KINDS)
def test_empty_selection_runs_through_the_workflow(kind) -> None:
    metadata = _crop().prediction_metadata()
    payload = _representations(metadata)[kind]

    row = _run(kind, payload, indices=[])

    assert tuple(row["own"].mask.shape) == (0, HEIGHT, WIDTH)
    assert tuple(row["own"].xyxy.shape) == (0, 4)


@pytest.mark.parametrize(
    "kind,expected",
    [
        ("instance_segmentation_view", "approximate"),
        ("instance_segmentation_prediction", "reference"),
        ("rle_instance_segmentation_prediction", "reference"),
    ],
)
def test_approximate_requests_report_what_was_computed(kind, expected) -> None:
    metadata = _crop().prediction_metadata()
    payload = _representations(metadata)[kind]

    row = _run(kind, payload, indices=[0, 1], fidelity="approximate")

    assert row["fidelity"] == expected
    # On an identity grid, the nearest expansion equals the reference.
    assert torch.equal(row["own"].mask, _masks()[[0, 1]])
