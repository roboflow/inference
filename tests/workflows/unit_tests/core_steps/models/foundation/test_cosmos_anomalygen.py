"""Unit tests for the Cosmos AnomalyGen v1 block."""

import numpy as np
import pytest
import supervision as sv
from pydantic import ValidationError

from inference.core.workflows.core_steps.common.entities import StepExecutionMode
from inference.core.workflows.core_steps.models.foundation.cosmos_anomalygen.v1 import (
    BlockManifest,
    CosmosAnomalyGenBlockV1,
    rasterize_placement_mask,
)

BASE = {
    "type": "roboflow_core/cosmos_anomalygen@v1",
    "name": "my_anomalygen_step",
    "image": "$inputs.image",
    "segmentation_mask": "$steps.model.predictions",
    "anomaly_type": "wood+crack",
}


def test_manifest_parses_with_defaults():
    result = BlockManifest.model_validate(BASE)
    assert result.anomaly_type == "wood+crack"
    assert result.model_version == "cosmos-anomalygen"
    assert result.guidance == 7.0
    assert result.num_steps == 35
    assert result.seed == 0


def test_manifest_accepts_selectors():
    result = BlockManifest.model_validate(
        {
            **BASE,
            "anomaly_type": "$inputs.anomaly_type",
            "model_version": "$inputs.model_version",
            "guidance": "$inputs.guidance",
        }
    )
    assert result.anomaly_type == "$inputs.anomaly_type"
    assert result.model_version == "$inputs.model_version"
    assert result.guidance == "$inputs.guidance"


def test_manifest_requires_segmentation_mask():
    payload = dict(BASE)
    del payload["segmentation_mask"]
    with pytest.raises(ValidationError):
        BlockManifest.model_validate(payload)


def test_manifest_declares_image_output():
    outputs = BlockManifest.describe_outputs()
    assert [o.name for o in outputs] == ["image"]


def test_remote_execution_raises():
    block = CosmosAnomalyGenBlockV1(
        api_key=None, step_execution_mode=StepExecutionMode.REMOTE
    )
    with pytest.raises(NotImplementedError):
        block.run(
            image=None,
            segmentation_mask=None,
            anomaly_type="wood+crack",
            model_version="cosmos-anomalygen",
            guidance=7.0,
            num_steps=35,
            seed=0,
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
