"""Unit tests for the Cosmos AnomalyGen v1 block."""

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.cosmos_anomalygen.v1 import (
    BlockManifest,
    CosmosAnomalyGenBlockV1,
    compute_visibility,
    rasterize_placement_mask,
)

BASE = {
    "type": "roboflow_core/cosmos_anomalygen@v1",
    "name": "my_anomalygen_step",
    "image": "$inputs.image",
    "segmentation_mask": "$steps.model.predictions",
    "anomaly_type": "wood+crack",
}


def test_manifest_parses_with_production_recipe_defaults():
    result = BlockManifest.model_validate(BASE)
    assert result.anomaly_type == "wood+crack"
    assert result.model_version == "cosmos-anomalygen"
    assert result.guidance == 1.5
    assert result.num_steps == 35
    assert result.seed == 0
    assert result.crop_ratio == 4.0
    assert result.poisson_blend is False


def test_manifest_accepts_selectors():
    result = BlockManifest.model_validate(
        {
            **BASE,
            "anomaly_type": "$inputs.anomaly_type",
            "model_version": "$inputs.model_version",
            "guidance": "$inputs.guidance",
            "crop_ratio": "$inputs.crop_ratio",
        }
    )
    assert result.anomaly_type == "$inputs.anomaly_type"
    assert result.model_version == "$inputs.model_version"
    assert result.guidance == "$inputs.guidance"
    assert result.crop_ratio == "$inputs.crop_ratio"


def test_manifest_requires_segmentation_mask():
    payload = dict(BASE)
    del payload["segmentation_mask"]
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(payload)


def test_manifest_declares_image_and_visibility_outputs():
    outputs = BlockManifest.describe_outputs()
    assert [o.name for o in outputs] == ["image", "visibility"]


def test_remote_execution_raises_at_init():
    with pytest.raises(NotImplementedError):
        CosmosAnomalyGenBlockV1(
            api_key=None, step_execution_mode=StepExecutionMode.REMOTE
        )


def test_rasterize_placement_mask_marks_masked_region_white():
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    mask = np.zeros((1, 10, 10), dtype=bool)
    mask[0, 2:6, 2:6] = True
    detections = sv.Detections(
        xyxy=np.array([[2.0, 2.0, 6.0, 6.0]]),
        mask=mask,
        class_id=np.array([0]),
    )

    result = rasterize_placement_mask(image=image, segmentation_mask=detections)

    assert result.shape == (10, 10)
    assert result[3, 3] == 255
    assert result[0, 0] == 0


def test_compute_visibility_measures_change_inside_mask_only():
    original = np.zeros((10, 10, 3), dtype=np.uint8)
    generated = original.copy()
    generated[2:6, 2:6] = 100  # change inside the mask
    generated[8, 8] = 255  # change outside the mask - must not count
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    visibility = compute_visibility(original=original, generated=generated, mask=mask)

    assert visibility == pytest.approx(100.0)


def test_compute_visibility_is_zero_for_unchanged_canvas():
    original = np.full((10, 10, 3), 50, dtype=np.uint8)
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:6, 2:6] = 255

    visibility = compute_visibility(
        original=original, generated=original.copy(), mask=mask
    )

    assert visibility == 0.0
