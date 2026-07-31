"""
Dependent-resources discovery tests for the SAM 3 Interactive block
(``roboflow_core/sam3_interactive@v1``) — the manifest exposes NO model
field; the block always runs the module-level ``SAM3_INTERACTIVE_MODEL_ID``.
"""

from inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive.v1 import (
    SAM3_INTERACTIVE_MODEL_ID,
)
from inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive.v1 import (
    BlockManifest as SegmentAnything3InteractiveV1Manifest,
)
from inference.core.workflows.prototypes.block import roboflow_platform_model


def test_sam3_interactive_model_id_constant_value() -> None:
    assert SAM3_INTERACTIVE_MODEL_ID == "sam3/sam3_interactive"


def test_sam3_interactive_v1_declares_constant_model_id() -> None:
    manifest = SegmentAnything3InteractiveV1Manifest.model_validate(
        {
            "type": "roboflow_core/sam3_interactive@v1",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id=SAM3_INTERACTIVE_MODEL_ID),
    ]


def test_sam3_interactive_v1_declares_constant_model_id_with_prompts() -> None:
    manifest = SegmentAnything3InteractiveV1Manifest.model_validate(
        {
            "type": "roboflow_core/sam3_interactive@v1",
            "name": "model",
            "images": "$inputs.image",
            "points": [{"x": 320, "y": 240, "positive": True}],
            "boxes": "$steps.detector.predictions",
        }
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(model_id="sam3/sam3_interactive"),
    ]


# ---------------------------------------------------------------------------
# SAM3_EXEC_MODE governs declared execution locality
# ---------------------------------------------------------------------------

import inference.core.workflows.core_steps.models.foundation.segment_anything3_interactive.v1 as sam3_interactive_module
import pytest

from inference.core.workflows.prototypes.block import ModelExecutionLocation


@pytest.mark.parametrize(
    "sam3_exec_mode,expected_location",
    [
        ("local", ModelExecutionLocation.ENVIRONMENT_DEFINED),
        ("remote", ModelExecutionLocation.REMOTE),
    ],
)
def test_sam3_interactive_declared_execution_location_follows_sam3_exec_mode(
    sam3_exec_mode, expected_location, monkeypatch
) -> None:
    monkeypatch.setattr(sam3_interactive_module, "SAM3_EXEC_MODE", sam3_exec_mode)
    manifest = SegmentAnything3InteractiveV1Manifest.model_validate(
        {
            "type": "roboflow_core/sam3_interactive@v1",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    (resource,) = manifest.discover_dependent_resources()

    assert resource.metadata.execution_location is expected_location
