"""SAM2 and SAM3 blocks on cached real model packages (opt-in).

Each test runs only when its backend library is installed and its package
directory is given; it never downloads. Otherwise it is skipped with the
missing piece named, and real SAM execution stays unverified:

    WORKFLOWS_V2_SAM2_PACKAGE   cached SAM2 package directory   needs ``sam2``
    WORKFLOWS_V2_SAM3_PACKAGE   cached SAM3 package directory   needs ``sam3``
    WORKFLOWS_V2_SAM_DEVICE     torch device, default "cuda" when available

The image is synthetic: a bright 120x120 square on a dark background. SAM2's
mask for a box around the square is checked against the square; SAM3 results
are checked for their contract, not for what SAM3 finds.
"""

import importlib.util
import os
from pathlib import Path
from typing import Any

import pytest
import torch
from roboflow_workflows.v2.blocks.models._shared.segment_anything import _mask_boxes
from roboflow_workflows.v2.blocks.models.segment_anything.v1 import SAM2_DEFAULT_MODEL_ID, SegmentAnything2Block
from roboflow_workflows.v2.blocks.models.sam3.v3 import SAM3_DEFAULT_MODEL_ID, SegmentAnything3Block
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.frames import FrameRef
from roboflow_workflows.v2.data.image import ImageData
from roboflow_workflows.v2.data.prediction_ops import CLASS_NAMES_KEY
from roboflow_workflows.v2.data.row_identity import row_record, with_row_identity
from roboflow_workflows.v2.engine.reactions.testing import block_call

from inference_models.models.base.object_detection import Detections

SQUARE_XYXY = [100, 60, 219, 179]


def _load(library: str, variable: str) -> Any:
    """The real model from the package directory, or a skip naming what is missing."""
    if importlib.util.find_spec(library) is None:
        pytest.skip(f"the {library} library is not installed")
    package = os.environ.get(variable)
    if not package or not Path(package).is_dir():
        pytest.skip(f"{variable} does not name a cached {library} package directory")

    from inference_models import AutoModel

    default = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(os.environ.get("WORKFLOWS_V2_SAM_DEVICE", default))
    model = AutoModel.from_pretrained(
        package, device=device, allow_untrusted_packages=True
    )

    return model


def _square_image() -> ImageData:
    pixels = torch.full((3, 240, 320), 30, dtype=torch.uint8)
    x1, y1, x2, y2 = SQUARE_XYXY
    pixels[:, y1 : y2 + 1, x1 : x2 + 1] = 220
    image = ImageData.from_tensor(pixels, image_id="square")

    return image


def _square_mask() -> torch.Tensor:
    mask = torch.zeros((240, 320), dtype=torch.bool)
    x1, y1, x2, y2 = SQUARE_XYXY
    mask[y1 : y2 + 1, x1 : x2 + 1] = True

    return mask


def _run(block: Any, **params: Any):
    with block_call(block):
        results = block.run(**params)

    return results


def test_real_sam2_segments_the_prompted_square_and_reuses_its_embedding() -> None:
    model = _load("sam2", "WORKFLOWS_V2_SAM2_PACKAGE")
    embedded = []
    embed_images = model.embed_images

    def counted(images, **options):
        embedded.append(len(images))
        return embed_images(images, **options)

    model.embed_images = counted
    block = SegmentAnything2Block(models={SAM2_DEFAULT_MODEL_ID: model})
    image = _square_image()
    prompt = with_row_identity(
        Detections(
            xyxy=torch.tensor([[95.0, 55.0, 224.0, 184.0]]),
            class_id=torch.tensor([0]),
            confidence=torch.tensor([0.9]),
            image_metadata={
                **FrameRef.from_image(image).prediction_metadata(),
                CLASS_NAMES_KEY: {0: "square"},
            },
        ),
        namespace="prompt",
    )

    (first,) = _run(block, image=Batch([image]), boxes=Batch([prompt]))
    (second,) = _run(block, image=Batch([image]))

    predictions = first["predictions"]
    assert len(predictions) == 1 and predictions.mask.dtype == torch.bool
    mask = predictions.mask[0].cpu()
    expected = _square_mask()
    iou = (mask & expected).sum() / (mask | expected).sum()
    assert float(iou) >= 0.8, f"mask IoU with the square is {float(iou):.3f}"
    assert torch.allclose(
        predictions.xyxy[0].cpu(),
        torch.tensor(SQUARE_XYXY, dtype=torch.float32),
        atol=6,
    )
    assert torch.equal(predictions.xyxy, _mask_boxes(predictions.mask))
    assert row_record(predictions).namespace == "prompt"
    assert predictions.image_metadata[CLASS_NAMES_KEY] == {0: "square"}
    assert len(second["predictions"]) == 1
    assert embedded == [1]


def test_real_sam3_text_prompt_results_keep_the_block_contract() -> None:
    model = _load("sam3", "WORKFLOWS_V2_SAM3_PACKAGE")
    block = SegmentAnything3Block(models={SAM3_DEFAULT_MODEL_ID: model})

    (result,) = _run(
        block,
        image=Batch([_square_image()]),
        class_names=["square", "circle"],
        confidence=0.3,
    )

    predictions = result["predictions"]
    assert predictions.mask.dtype == torch.bool
    assert tuple(predictions.mask.shape[1:]) == (240, 320)
    assert set(predictions.class_id.tolist()) <= {0, 1}
    assert bool((predictions.confidence >= 0.3).all())
    assert bool(predictions.mask.flatten(1).any(dim=1).all())
    assert torch.equal(predictions.xyxy, _mask_boxes(predictions.mask))
    assert predictions.image_metadata[CLASS_NAMES_KEY] == {0: "square", 1: "circle"}
    assert torch.equal(result["segmentation"].full_res().mask, predictions.mask)
