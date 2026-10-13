"""Decision 026: every V2 RLE decode validates the counts before the native decode.

pycocotools can decode short counts without an error into an invalid mask with
undefined uncovered pixels. Each case below has legal-looking input for an 8x8
image (64 pixels)::

    valid       4x4 square at rows/columns 2..5      covers 64
    short       b"1"   one background run            covers 1
    oversized   b"Q2"  one background run            covers 65
    bad byte    b"1!"  "!" is below the "0" base     syntax error
    dangling    b"1P"  "P" sets the continuation bit syntax error

Malformed counts must fail with a ``ContractError`` naming the original row
before the native decoder is called. The decoder is replaced by a sentinel in
those tests, so a regression cannot reach native code.
"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch
from pycocotools import mask as mask_utils
from roboflow_workflows.v2.blocks.geometry.overlap import (
    non_max_merge,
    non_max_suppression,
)
from roboflow_workflows.v2.data import rle
from roboflow_workflows.v2.data.coordinates import prediction_to_frame
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import FrameMapping, ImageData
from roboflow_workflows.v2.data.masks import providers
from roboflow_workflows.v2.data.masks.providers import RLEMaskProvider
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
    RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.rle import check_rle_counts, decode_rle_masks

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.types import InstancesRLEMasks

SIZE_HW = (8, 8)
SQUARE = np.zeros(SIZE_HW, dtype=bool)
SQUARE[2:6, 2:6] = True
VALID = mask_utils.encode(np.asfortranarray(SQUARE.astype(np.uint8)))["counts"]
MALFORMED = {
    "short": (b"1", "counts cover 1 pixels; image_size (8, 8) needs 64"),
    "oversized": (b"Q2", "counts cover 65 pixels; image_size (8, 8) needs 64"),
    "bad_byte": (b"1!", "byte 33 at 1 is not a compressed COCO count"),
    "dangling": (b"1P", "counts end inside a run length"),
}
IMAGE = ImageData.from_tensor(
    torch.full((3, *SIZE_HW), 100, dtype=torch.uint8), image_id="rle-image"
)
V2_SOURCE = Path(rle.__file__).resolve().parents[1]


@pytest.fixture
def native_decode_forbidden(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("malformed counts reached the native RLE decoder")

    monkeypatch.setattr(rle, "coco_rle_masks_to_numpy_mask", forbidden)
    monkeypatch.setattr(providers, "coco_rle_masks_to_torch_mask", forbidden)


def predictions(counts, *, box=(0, 0, 8, 8)):
    rows = len(counts)
    prediction = InstanceDetections(
        xyxy=torch.tensor([box] * rows, dtype=torch.float32),
        class_id=torch.zeros(rows, dtype=torch.int64),
        confidence=torch.linspace(0.9, 0.8, rows),
        mask=InstancesRLEMasks(image_size=SIZE_HW, masks=list(counts)),
        image_metadata={**IMAGE.prediction_metadata(), "class_names": {0: "object"}},
    )

    return prediction


def through_public_envelopes(prediction):
    # Both native kinds admit the payload; JSON round trip as a caller sends it.
    for kind in (
        INSTANCE_SEGMENTATION_PREDICTION_KIND,
        RLE_INSTANCE_SEGMENTATION_PREDICTION_KIND,
    ):
        kind.check(prediction)
    wire = json.loads(
        json.dumps(INSTANCE_SEGMENTATION_PREDICTION_KIND.to_serialized(prediction))
    )
    restored = INSTANCE_SEGMENTATION_PREDICTION_KIND.to_payload(wire)
    INSTANCE_SEGMENTATION_PREDICTION_KIND.check(restored)

    return restored


def into_root_shifted_by_one():
    mapping = FrameMapping(
        frame_id=IMAGE.image_id, frame_size_hw=SIZE_HW, offset_xy=(1.0, 1.0)
    )

    return dict(mapping=mapping, frame=FrameRef.from_image(IMAGE))


def test_valid_counts_decode_to_the_native_mask():
    restored = through_public_envelopes(predictions([VALID, VALID]))

    decoded = decode_rle_masks(restored.mask)

    assert decoded.dtype == np.bool_
    assert decoded.shape == (2, *SIZE_HW)
    assert np.array_equal(decoded[0], SQUARE) and np.array_equal(decoded[1], SQUARE)


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_malformed_counts_are_rejected_before_the_native_decode(
    case, native_decode_forbidden
):
    counts, message = MALFORMED[case]
    restored = through_public_envelopes(predictions([VALID, counts]))

    with pytest.raises(ContractError, match=re.escape(f"RLE row 1: {message}")):
        decode_rle_masks(restored.mask)


def test_counts_must_be_ascii_compressed_bytes():
    with pytest.raises(ContractError, match="RLE row 3: counts are not ASCII"):
        check_rle_counts("1é", size_hw=SIZE_HW, row=3)
    with pytest.raises(ContractError, match="counts must be bytes or str, got list"):
        check_rle_counts([64], size_hw=SIZE_HW, row=0)
    with pytest.raises(ContractError, match="RLE row 0: negative run length -16"):
        check_rle_counts(b"@", size_hw=SIZE_HW, row=0)


def test_selected_rows_validate_and_decode_only_those_rows():
    short = MALFORMED["short"][0]
    mask = predictions([short, VALID, short]).mask

    decoded = decode_rle_masks(mask, rows=[1])

    assert decoded.shape == (1, *SIZE_HW)
    assert np.array_equal(decoded[0], SQUARE)
    with pytest.raises(ContractError, match="RLE row 2: counts cover 1 pixels"):
        decode_rle_masks(mask, rows=[1, 2])


def test_provider_decodes_valid_counts_on_the_cpu():
    provider = RLEMaskProvider(through_public_envelopes(predictions([VALID])))

    mask = provider.binary(fidelity="reference").predictions.mask

    assert mask.device.type == "cpu" and mask.dtype == torch.bool
    assert torch.equal(mask[0], torch.from_numpy(SQUARE))


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_provider_rejects_malformed_counts(case, native_decode_forbidden):
    counts, message = MALFORMED[case]
    provider = RLEMaskProvider(through_public_envelopes(predictions([VALID, counts])))

    with pytest.raises(ContractError, match=re.escape(f"RLE row 1: {message}")):
        provider.binary(fidelity="reference")
    assert not provider.is_decoded


def test_frame_mapping_moves_valid_rle_masks():
    moved = prediction_to_frame(
        through_public_envelopes(predictions([VALID])), **into_root_shifted_by_one()
    )

    expected = np.zeros(SIZE_HW, dtype=bool)
    expected[3:7, 3:7] = True
    assert np.array_equal(decode_rle_masks(moved.mask)[0], expected)


@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_frame_mapping_rejects_malformed_rle_masks(case, native_decode_forbidden):
    counts, message = MALFORMED[case]
    restored = through_public_envelopes(predictions([VALID, counts]))

    with pytest.raises(ContractError, match=re.escape(f"RLE row 1: {message}")):
        prediction_to_frame(restored, **into_root_shifted_by_one())


@pytest.mark.parametrize("operation", [non_max_suppression, non_max_merge])
def test_mask_overlap_of_valid_rle_masks(operation):
    duplicates = through_public_envelopes(predictions([VALID, VALID], box=(2, 2, 6, 6)))

    kept = operation(duplicates, threshold=0.5, use_masks=True)

    assert len(kept.xyxy) == 1
    assert np.array_equal(decode_rle_masks(kept.mask)[0], SQUARE)


@pytest.mark.parametrize("operation", [non_max_suppression, non_max_merge])
@pytest.mark.parametrize("case", sorted(MALFORMED))
def test_mask_overlap_rejects_malformed_rle_masks(
    operation, case, native_decode_forbidden
):
    counts, message = MALFORMED[case]
    restored = through_public_envelopes(predictions([VALID, counts], box=(2, 2, 6, 6)))

    with pytest.raises(ContractError, match=re.escape(f"RLE row 1: {message}")):
        operation(restored, threshold=0.5, use_masks=True)


def test_only_the_shared_helper_and_provider_call_the_native_decoders():
    # Direct native decodes bypassed validation before decision 026.
    callers = sorted(
        str(path.relative_to(V2_SOURCE))
        for path in V2_SOURCE.rglob("*.py")
        if "coco_rle_masks_to_" in path.read_text()
    )

    assert callers == ["data/masks/providers.py", "data/rle.py"]
