"""Behavioural tests for the legacy Gaze stub model."""

import pytest

from inference.core.exceptions import FeatureDeprecatedError
from inference.models.gaze.gaze import Gaze
from inference.models.gaze.gaze_inference_models import InferenceModelsGazeAdapter


def test_gaze_stub_raises_feature_deprecated_error_on_init() -> None:
    with pytest.raises(FeatureDeprecatedError) as captured:
        Gaze()

    assert captured.value.feature == "Gaze (L2CS-Net) model"


def test_gaze_stub_raises_regardless_of_kwargs() -> None:
    with pytest.raises(FeatureDeprecatedError):
        Gaze("model-id", api_key="anything", arbitrary=True)


def test_inference_models_gaze_adapter_raises_feature_deprecated_error_on_init() -> None:
    with pytest.raises(FeatureDeprecatedError) as captured:
        InferenceModelsGazeAdapter()

    assert "inference_models adapter" in captured.value.feature


def test_gaze_stub_resolves_in_roboflow_model_types_registry() -> None:
    # If CORE_MODEL_GAZE_ENABLED is True (default), the registry must still
    # resolve ("gaze", "l2cs") to the stub so model-id lookups raise
    # FeatureDeprecatedError rather than KeyError.
    from inference.core.env import CORE_MODEL_GAZE_ENABLED
    from inference.models.utils import ROBOFLOW_MODEL_TYPES

    if not CORE_MODEL_GAZE_ENABLED:
        pytest.skip("CORE_MODEL_GAZE_ENABLED is False — registry entry is opt-in.")

    assert ROBOFLOW_MODEL_TYPES.get(("gaze", "l2cs")) is Gaze
