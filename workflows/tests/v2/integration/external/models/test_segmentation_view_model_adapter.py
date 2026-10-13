"""The development YOLOv8 view adapter on a cached real model package (opt-in).

Runs only when the cached ``yolov8n-seg-640`` ONNX package directory exists
(``SEGMENTATION_VIEW_PACKAGE`` overrides its path); it never downloads. The
synthetic cases are ``integration/data/test_segmentation_view_model_adapter.py``.
"""

import os
from pathlib import Path

import pytest
import torch

from tests.v2.support.data.segmentation_adapter import (
    SelectionSettings,
    YoloSegmentationViewAdapter,
    check_selection,
)

DEFAULT_PACKAGE = Path(
    "/tmp/cache/models-cache/v2-coco-dataset-vdnr1-2-1c89587e04f6853221f2795bd27beb16"
    "/cd523840d31f1b244a4a8e77a5bde159"
)


def _package_dir() -> Path:
    package = Path(os.environ.get("SEGMENTATION_VIEW_PACKAGE", DEFAULT_PACKAGE))

    return package


@pytest.mark.skipif(
    not _package_dir().is_dir(), reason="cached YOLO seg package absent"
)
def test_real_package_matches_its_dense_post_process() -> None:
    from inference_models import AutoModel

    model = AutoModel.from_pretrained(
        str(_package_dir()),
        device=torch.device("cpu"),
        onnx_execution_providers=["CPUExecutionProvider"],
        allow_untrusted_packages=True,
    )
    adapter = YoloSegmentationViewAdapter(
        model, settings=SelectionSettings(confidence=0.05)
    )
    generator = torch.Generator().manual_seed(0)
    image = torch.randint(0, 256, (3, 333, 517), dtype=torch.uint8, generator=generator)

    batch, metadata = adapter.pre_process(image)
    raw = adapter.forward(batch)
    view = adapter.view(adapter.select(raw, metadata))
    reference = adapter.dense_reference(raw, metadata)

    result = check_selection(view, reference)
    assert all(result[key] for key in result if key.endswith("_equal")), result
