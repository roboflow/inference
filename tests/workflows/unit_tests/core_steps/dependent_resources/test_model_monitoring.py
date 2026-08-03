"""
Dependent-resources discovery tests for the Model Monitoring Inference
Aggregator sink (``roboflow_core/model_monitoring_inference_aggregator@v1``).

``model_id`` is a selector-only field (``Selector(kind=[ROBOFLOW_MODEL_ID_KIND])``
with no literal ``str`` arm). The block references the platform model entity
for monitoring without pulling weights, so the declared dependency uses
``ModelRequiredAction.ACCESS``.
"""

import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.sinks.roboflow.model_monitoring_inference_aggregator.v1 import (
    BlockManifest as ModelMonitoringV1Manifest,
)
from inference.core.workflows.prototypes.block import (
    ModelRequiredAction,
    roboflow_platform_model,
)


def _build_payload(**overrides) -> dict:
    payload = {
        "type": "roboflow_core/model_monitoring_inference_aggregator@v1",
        "name": "monitor",
        "predictions": "$steps.model.predictions",
        "model_id": "$inputs.model",
        "unique_aggregator_key": "aggregator-1",
    }
    payload.update(overrides)
    return payload


def test_model_monitoring_v1_declares_access_only_model_dependency() -> None:
    manifest = ModelMonitoringV1Manifest.model_validate(_build_payload())

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(
            model_id="$inputs.model",
            required_action=ModelRequiredAction.ACCESS,
        ),
    ]


def test_model_monitoring_v1_returns_step_output_selector_verbatim() -> None:
    manifest = ModelMonitoringV1Manifest.model_validate(
        _build_payload(model_id="$steps.detector.model_id")
    )

    assert manifest.discover_dependent_resources() == [
        roboflow_platform_model(
            model_id="$steps.detector.model_id",
            required_action=ModelRequiredAction.ACCESS,
        ),
    ]


def test_model_monitoring_v1_rejects_literal_model_id() -> None:
    # Documents that the field is selector-only: a plain "project/version"
    # literal does not match the selector pattern and fails validation.
    with pytest.raises(ValidationError):
        ModelMonitoringV1Manifest.model_validate(
            _build_payload(model_id="my_project/3")
        )
