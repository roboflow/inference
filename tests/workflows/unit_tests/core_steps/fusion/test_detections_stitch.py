from typing import Union

import pytest

from inference.core.workflows.core_steps.fusion.detections_stitch.v1 import (
    BlockManifest,
)


@pytest.mark.parametrize(
    "overlap_filtering_strategy",
    ["none", "nms", "nmm", "$inputs.some"],
)
@pytest.mark.parametrize(
    "iou_threshold",
    [0.5, "$inputs.some"],
)
def test_detections_stitch_v1_manifest_parsing_when_input_valid(
    overlap_filtering_strategy: str,
    iou_threshold: Union[float, str],
) -> None:
    raw_manifest = {
        "type": "roboflow_core/detections_stitch@v1",
        "name": "stitch",
        "reference_image": "$inputs.image",
        "predictions": "$steps.model.predictions",
        "overlap_filtering_strategy": overlap_filtering_strategy,
        "iou_threshold": iou_threshold,
    }

    # when
    result = BlockManifest.model_validate(raw_manifest)

    # then
    assert result == BlockManifest(
        type="roboflow_core/detections_stitch@v1",
        name="stitch",
        reference_image="$inputs.image",
        predictions="$steps.model.predictions",
        overlap_filtering_strategy=overlap_filtering_strategy,
        iou_threshold=iou_threshold,
    )


def test_detections_stitch_v1_manifest_parsing_when_overlap_mode_invalid() -> None:
    raw_manifest = {
        "type": "roboflow_core/detections_stitch@v1",
        "name": "stitch",
        "reference_image": "$inputs.image",
        "predictions": "$steps.model.predictions",
        "overlap_filtering_strategy": "invalid",
        "iou_threshold": 0.5,
    }

    # when
    with pytest.raises(ValueError):
        _ = BlockManifest.model_validate(raw_manifest)
