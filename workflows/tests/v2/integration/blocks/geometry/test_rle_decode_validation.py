"""Decision 026: compiled workflows reject malformed RLE masks before decoding.

Each consumer decodes ``InstancesRLEMasks`` through ``data/rle.py``, which
checks the compressed counts first. Cases on an 8x8 image (64 pixels)::

    valid       4x4 square at rows/columns 2..5      covers 64
    short       b"1"   one background run            covers 1
    oversized   b"Q2"  one background run            covers 65
    bad byte    b"1!"  "!" is below the "0" base     syntax error

    fixture or caller JSON ──> dynamic crop fade   ──┐
                           ──> bounding rect       ──┤ data/rle.py check
                           ──> stitch none/nms/nmm ──┤ then native decode
    crop-frame output      ──> root conversion     ──┘

Malformed rows fail the step with a ``ContractError`` cause naming the row;
the native decoder is replaced by a sentinel in those tests. Rows that are not
decoded, such as an empty crop box, are not checked.
"""

import json
from typing import Any, Dict, List

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from pydantic import Field
from roboflow_workflows.v2.data import rle
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.image_kinds import IMAGE_KIND
from roboflow_workflows.v2.data.masks import providers
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.row_identity import with_row_identity
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.declaration import Block, BlockParams, Output, Ref
from roboflow_workflows.v2.engine.errors import WorkflowExecutionError

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks
from tests.v2.integration.blocks.geometry.deterministic import regions_catalogue

SIZE_HW = (8, 8)
FULL_BOX = [0.0, 0.0, 8.0, 8.0]
EMPTY_BOX = [3.0, 3.0, 3.0, 3.0]
MALFORMED = {
    "short": (b"1", "counts cover 1 pixels; image_size (8, 8) needs 64"),
    "oversized": (b"Q2", "counts cover 65 pixels; image_size (8, 8) needs 64"),
    "bad_byte": (b"1!", "byte 33 at 1 is not a compressed COCO count"),
}


def centre_square(size_hw):
    # The middle half along each axis: rows/columns 2..5 on 8x8.
    height, width = size_hw
    square = np.zeros(size_hw, dtype=bool)
    square[height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = True

    return square


def counts_of(case, *, size_hw):
    if case == "valid":
        dense = np.asfortranarray(centre_square(size_hw).astype(np.uint8))
        valid = mask_utils.encode(dense)["counts"]
        return valid

    return MALFORMED[case][0]


def rle_prediction(cases, *, boxes, image):
    rows = len(cases)
    prediction = InstanceDetections(
        xyxy=torch.tensor(boxes, dtype=torch.float32),
        class_id=torch.zeros(rows, dtype=torch.int64),
        confidence=torch.linspace(0.9, 0.8, rows),
        mask=InstancesRLEMasks(
            image_size=image.size_hw,
            masks=[counts_of(case, size_hw=image.size_hw) for case in cases],
        ),
        image_metadata={**image.prediction_metadata(), "class_names": {0: "object"}},
    )

    identified = with_row_identity(prediction, namespace="rle-fixture")

    return identified


class RLEFixtureBlock(Block):
    """Instance segmentation with one compressed RLE row per entry of ``cases``."""

    type = "test/rle_fixture@v1"
    outputs = {
        "predictions": Output(
            INSTANCE_SEGMENTATION_PREDICTION_KIND, source="image", expand="rows"
        )
    }

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="Image the masks describe.")
        cases: List[str] = Field(description="Mask case of each row.")
        boxes: List[List[float]] = Field(description="Box of each row.")

    def run(
        self, *, image: ImageData, cases: List[str], boxes: List[List[float]]
    ) -> Dict[str, Any]:
        prediction = rle_prediction(cases, boxes=boxes, image=image)

        return {"predictions": Batch.of([prediction])}


def image_of_100():
    image = ImageData.from_tensor(
        torch.full((3, *SIZE_HW), 100, dtype=torch.uint8), image_id="rle-image"
    )

    return image


def produce(cases, *, boxes=None, image="$inputs.image"):
    step = {
        "type": RLEFixtureBlock.type,
        "name": "produce",
        "image": image,
        "cases": cases,
        "boxes": boxes or [FULL_BOX] * len(cases),
    }

    return step


def crop_fade(predictions="$steps.produce.predictions"):
    step = {
        "type": "roboflow_core/dynamic_crop@v1",
        "name": "consume",
        "image": "$inputs.image",
        "predictions": predictions,
        "mask_opacity": 1.0,
    }

    return step


def bounding_rect():
    step = {
        "type": "roboflow_core/bounding_rect@v1",
        "name": "consume",
        "predictions": "$steps.produce.predictions",
    }

    return step


def stitch(strategy):
    step = {
        "type": "roboflow_core/detections_stitch@v1",
        "name": "consume",
        "reference_image": "$inputs.image",
        "predictions": "$steps.produce.predictions",
        "overlap_filtering_strategy": strategy,
        "overlap_by": "mask",
    }

    return step


def run(steps, outputs, *, inputs=None, extra_inputs=()):
    result = run_session(steps, outputs, inputs=inputs, extra_inputs=extra_inputs)

    return result.outputs.data


def run_session(steps, outputs, *, inputs=None, extra_inputs=()):
    workflow = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowImage", "name": "image"}, *extra_inputs],
        "steps": steps,
        "outputs": [
            {"type": "JsonField", "name": name, "selector": selector}
            for name, selector in outputs.items()
        ],
    }
    catalogue = regions_catalogue().with_blocks([RLEFixtureBlock])
    plan = compile_workflow(workflow, catalogue=catalogue)
    result = plan.create_session().run(inputs or {"image": [image_of_100()]})

    return result


@pytest.fixture
def native_decode_forbidden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("malformed counts reached the native RLE decoder")

    monkeypatch.setattr(rle, "coco_rle_masks_to_numpy_mask", forbidden)
    monkeypatch.setattr(providers, "coco_rle_masks_to_torch_mask", forbidden)


def assert_rejected(raised, *, row, case, location="$steps.consume"):
    # The error names the step or output; its cause names the original row.
    error = raised.value
    assert location in str(error)
    cause = error.__cause__
    assert isinstance(cause, ContractError)
    assert f"RLE row {row}: {MALFORMED[case][1]}" in str(cause)


def test_crop_fade_keeps_the_valid_mask_pixels():
    raw = run([produce(["valid"]), crop_fade()], {"crops": "$steps.consume.crops"})

    # Crops nest per image, per produced prediction, then one slot per row.
    (crop,) = raw["crops"][0][0]
    inside = torch.from_numpy(centre_square(SIZE_HW)).expand(3, *SIZE_HW)
    assert torch.equal(crop.tensor_image, torch.where(inside, 100, 0).byte())
    assert int(crop.tensor_image.sum()) == 4800


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_crop_fade_rejects_malformed_masks(case, native_decode_forbidden):
    with pytest.raises(WorkflowExecutionError) as raised:
        run([produce([case]), crop_fade()], {"crops": "$steps.consume.crops"})

    assert_rejected(raised, row=0, case=case)


CALLER_PREDICTIONS = {
    "type": "WorkflowBatchInput",
    "name": "predictions",
    "kind": ["instance_segmentation_prediction"],
}


def crop_fade_of_caller_json(case):
    # The caller sends the native envelope as JSON; no producer step.
    image = image_of_100()
    native = rle_prediction([case], boxes=[FULL_BOX], image=image)
    wire = json.loads(
        json.dumps(INSTANCE_SEGMENTATION_PREDICTION_KIND.to_serialized(native))
    )
    raw = run(
        [crop_fade(predictions="$inputs.predictions")],
        {"crops": "$steps.consume.crops"},
        inputs={"image": [image], "predictions": [wire]},
        extra_inputs=[CALLER_PREDICTIONS],
    )

    return raw


def test_crop_fade_of_a_valid_caller_json_mask():
    raw = crop_fade_of_caller_json("valid")

    (crop,) = raw["crops"][0]
    assert int(crop.tensor_image.sum()) == 4800


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_crop_fade_rejects_malformed_caller_json_masks(case, native_decode_forbidden):
    with pytest.raises(WorkflowExecutionError) as raised:
        crop_fade_of_caller_json(case)

    assert_rejected(raised, row=0, case=case)


def test_crop_fade_never_checks_the_mask_of_an_empty_box(monkeypatch):
    decoded = []
    native_decode = rle.coco_rle_masks_to_numpy_mask

    def recording(masks):
        decoded.extend(masks.masks)
        return native_decode(masks)

    monkeypatch.setattr(rle, "coco_rle_masks_to_numpy_mask", recording)
    raw = run(
        [produce(["valid", "short"], boxes=[FULL_BOX, EMPTY_BOX]), crop_fade()],
        {"crops": "$steps.consume.crops", "rows": "$steps.consume.predictions"},
    )

    first, second = raw["crops"][0][0]
    assert int(first.tensor_image.sum()) == 4800
    assert second is None and raw["rows"][0][0][1] is None
    assert MALFORMED["short"][0] not in decoded


def test_bounding_rect_of_a_valid_mask():
    raw = run(
        [produce(["valid"]), bounding_rect()],
        {"predictions": "$steps.consume.predictions"},
    )

    (prediction,) = raw["predictions"][0]
    assert prediction.xyxy.tolist() == [[2.0, 2.0, 5.0, 5.0]]
    assert prediction.bboxes_metadata[0]["rect"] == [[2, 2], [5, 2], [5, 5], [2, 5]]


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_bounding_rect_rejects_malformed_masks(case, native_decode_forbidden):
    with pytest.raises(WorkflowExecutionError) as raised:
        run(
            [produce(["valid", case]), bounding_rect()],
            {"predictions": "$steps.consume.predictions"},
        )

    assert_rejected(raised, row=1, case=case)


@pytest.mark.parametrize("strategy, rows", [("none", 2), ("nms", 1), ("nmm", 1)])
def test_stitch_of_valid_masks(strategy, rows):
    raw = run(
        [produce(["valid", "valid"]), stitch(strategy)],
        {"predictions": "$steps.consume.predictions"},
    )

    prediction = raw["predictions"][0]
    assert len(prediction.xyxy) == rows
    assert prediction.mask.image_size == SIZE_HW


@pytest.mark.parametrize("strategy", ["none", "nms", "nmm"])
@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_stitch_rejects_malformed_masks(strategy, case, native_decode_forbidden):
    with pytest.raises(WorkflowExecutionError) as raised:
        run(
            [produce(["valid", case]), stitch(strategy)],
            {"predictions": "$steps.consume.predictions"},
        )

    assert_rejected(raised, row=1, case=case)


def crop_frame_masks(case):
    # A 4x4 crop at offset (3, 3); the fixture predicts in the crop's frame.
    prepare = {
        "type": "roboflow_core/absolute_static_crop@v1",
        "name": "prepare",
        "image": "$inputs.image",
        "x_center": 5,
        "y_center": 5,
        "width": 4,
        "height": 4,
    }
    result = run_session(
        [
            prepare,
            produce([case], boxes=[[0.0, 0.0, 4.0, 4.0]], image="$steps.prepare.crops"),
        ],
        {"predictions": "$steps.produce.predictions"},
    )

    return result


def test_crop_frame_masks_convert_to_the_root():
    (row,) = crop_frame_masks("valid").rows()

    (prediction,) = row["predictions"]
    assert prediction.mask.image_size == SIZE_HW
    root_mask = mask_utils.decode(
        {"size": list(SIZE_HW), "counts": prediction.mask.masks[0]}
    )
    expected = np.zeros(SIZE_HW, dtype=np.uint8)
    expected[4:6, 4:6] = 1
    assert np.array_equal(root_mask, expected)


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_crop_frame_output_rejects_malformed_masks(case, native_decode_forbidden):
    # The 4x4 crop needs 16 pixels, so only the start of the message differs.
    result = crop_frame_masks(case)

    with pytest.raises(WorkflowExecutionError) as raised:
        result.rows()

    error = raised.value
    assert "predictions" in str(error) and "convert_output" in str(error)
    assert isinstance(error.__cause__, ContractError)
    assert str(error.__cause__).startswith("RLE row 0: ")
