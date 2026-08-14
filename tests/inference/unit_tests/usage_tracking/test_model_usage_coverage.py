"""Guards that model families reimplementing ``infer()`` still report usage.

``@usage_collector("model")`` normally rides along on ``BaseInference.infer``.
These families override ``infer()`` and never call ``super()``, so the decorator
has to be applied to each override; without it they emit no model-category row
and disappear from per-model telemetry entirely.
"""

import inspect

import pytest

# (module, class) pairs whose ``infer()`` bypasses ``BaseInference.infer``.
MODELS_OVERRIDING_INFER = [
    ("inference.models.doctr.doctr_model", "DocTR"),
    (
        "inference.models.doctr.doctr_model_inference_models",
        "InferenceModelsDocTRAdapter",
    ),
    ("inference.models.grounding_dino.grounding_dino", "GroundingDINO"),
    (
        "inference.models.grounding_dino.grounding_dino_inference_models",
        "InferenceModelsGroundingDINOAdapter",
    ),
    ("inference.models.yolo_world.yolo_world", "YOLOWorld"),
    ("inference.models.owlv2.owlv2", "OwlV2"),
    ("inference.models.owlv2.owlv2", "SerializedOwlV2"),
    ("inference.models.owlv2.owlv2_inference_models", "InferenceModelsOwlV2Adapter"),
    (
        "inference.models.owlv2.rf_instant_inference_models",
        "InferenceModelsRFInstantModelAdapter",
    ),
]


def _is_usage_collected(func) -> bool:
    # functools.wraps hides the wrapper, so read the unfollowed signature and
    # look for the keyword-only arguments the decorator injects.
    parameters = inspect.signature(func, follow_wrapped=False).parameters
    return "usage_api_key" in parameters and "usage_billable" in parameters


@pytest.mark.parametrize("module_path, class_name", MODELS_OVERRIDING_INFER)
def test_model_overriding_infer_is_usage_collected(module_path, class_name):
    module = pytest.importorskip(module_path)
    model_class = getattr(module, class_name)

    assert _is_usage_collected(
        model_class.infer
    ), f"{class_name}.infer() must be decorated with @usage_collector('model')"


def test_detection_helper_rejects_undecorated_function():
    # Without this the parametrized test above would pass vacuously.
    def infer(self, image, **kwargs):
        return None

    assert not _is_usage_collected(infer)
