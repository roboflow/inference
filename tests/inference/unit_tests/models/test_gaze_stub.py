"""Behavioural tests for the legacy Gaze stub model."""

import pytest

from inference.core.exceptions import FeatureDeprecatedError
from inference.models.gaze.gaze import Gaze
from inference.models.gaze.gaze_inference_models import InferenceModelsGazeAdapter


@pytest.mark.parametrize(
    "stub_cls",
    [Gaze, InferenceModelsGazeAdapter],
    ids=["legacy-Gaze", "inference-models-adapter"],
)
def test_gaze_stubs_raise_feature_deprecated_error_on_init(stub_cls) -> None:
    with pytest.raises(FeatureDeprecatedError):
        stub_cls()


@pytest.mark.parametrize(
    "stub_cls",
    [Gaze, InferenceModelsGazeAdapter],
    ids=["legacy-Gaze", "inference-models-adapter"],
)
def test_gaze_stubs_raise_regardless_of_kwargs(stub_cls) -> None:
    with pytest.raises(FeatureDeprecatedError):
        stub_cls("model-id", api_key="anything", arbitrary=True)


def test_gaze_l2cs_registry_entry_resolves_to_a_stub_that_raises_feature_deprecated() -> None:
    """The registry entry under ("gaze", "l2cs") must still resolve so model-id
    lookups raise FeatureDeprecatedError rather than KeyError. Which specific
    stub class (legacy Gaze vs InferenceModelsGazeAdapter) wins depends on the
    order of registration in `inference/models/utils.py`; both are valid."""
    from inference.core.env import CORE_MODEL_GAZE_ENABLED
    from inference.models.utils import ROBOFLOW_MODEL_TYPES

    if not CORE_MODEL_GAZE_ENABLED:
        pytest.skip("CORE_MODEL_GAZE_ENABLED is False — registry entry is opt-in.")

    registered = ROBOFLOW_MODEL_TYPES.get(("gaze", "l2cs"))
    assert registered is not None, "Registry entry for ('gaze', 'l2cs') must survive."
    assert registered in {Gaze, InferenceModelsGazeAdapter}

    with pytest.raises(FeatureDeprecatedError):
        registered()
