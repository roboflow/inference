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


def test_sam3_interactive_declares_environment_defined_execution_when_local(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sam3_interactive_module, "SAM3_EXEC_MODE", "local")
    manifest = SegmentAnything3InteractiveV1Manifest.model_validate(
        {
            "type": "roboflow_core/sam3_interactive@v1",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    (resource,) = manifest.discover_dependent_resources()

    assert (
        resource.metadata.execution_location
        is ModelExecutionLocation.ENVIRONMENT_DEFINED
    )


def test_sam3_interactive_declares_nothing_under_proxy_execution_mode(
    monkeypatch,
) -> None:
    monkeypatch.setattr(sam3_interactive_module, "SAM3_EXEC_MODE", "remote")
    manifest = SegmentAnything3InteractiveV1Manifest.model_validate(
        {
            "type": "roboflow_core/sam3_interactive@v1",
            "name": "model",
            "images": "$inputs.image",
        }
    )

    # The proxy runs its own fixed SAM3 server-side — nothing to declare.
    assert manifest.discover_dependent_resources() == []
