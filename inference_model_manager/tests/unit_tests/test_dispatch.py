"""Unit tests for dispatch module — action resolution and listing."""

from inference_model_manager.dispatch import list_actions_by_mro_names
from inference_model_manager.registry_defaults import lazy_register_by_names


def test_list_actions_by_mro_names_object_detection():
    """ObjectDetectionModel MRO names should return 'infer' action."""
    lazy_register_by_names(["ObjectDetectionModel"])
    actions = list_actions_by_mro_names(["ObjectDetectionModel"])
    assert "infer" in actions
    assert actions["infer"]["default"] is True
    assert "images" in actions["infer"]["params"]


def test_list_actions_by_mro_names_unknown_class():
    """Unknown class names should return empty dict."""
    actions = list_actions_by_mro_names(["CompletelyUnknownModel"])
    assert actions == {}


def test_list_actions_by_mro_names_walks_mro():
    """Should match on any ancestor in the MRO list."""
    lazy_register_by_names(["ObjectDetectionModel"])
    actions = list_actions_by_mro_names(
        [
            "YOLOv8ForObjectDetectionTorchScript",
            "ObjectDetectionModel",
            "object",
        ]
    )
    assert "infer" in actions


def test_invoke_action_applies_param_aliases():
    from inference_model_manager.dispatch import invoke_action
    from inference_model_manager.registry_defaults import registry

    class AliasedModel:
        def ask(self, question):
            return f"answer:{question}"

    registry.register(
        AliasedModel,
        "query",
        method="ask",
        default=True,
        params={"prompt": {"type": "str", "required": True}},
        validator=lambda kw: kw,
        serializer=lambda out, m: {"text": out},
        response_type="roboflow-text-v1",
        param_aliases={"prompt": "question"},
    )
    result = invoke_action(AliasedModel(), action="query", prompt="hi")
    assert result == "answer:hi"


def _register_capturing_model(declare_flags=True):
    from inference_model_manager.registry_defaults import (
        PRE_PROCESSING_OVERRIDE_FIELDS,
        registry,
    )

    params = {"images": {"type": "image", "required": True}}
    if declare_flags:
        params.update(
            {
                flag: {"type": "bool", "required": False}
                for flag in PRE_PROCESSING_OVERRIDE_FIELDS
            }
        )

    class CapturingModel:
        def __init__(self):
            self.received = None

        def infer(self, images, **kwargs):
            self.received = kwargs
            return []

    registry.register(
        CapturingModel,
        "infer",
        default=True,
        params=params,
        validator=lambda kw: kw,
        serializer=lambda out, m: {"predictions": out},
        response_type="roboflow-object-detection-compact-v1",
    )
    return CapturingModel()


def test_invoke_action_builds_pre_processing_overrides_from_legacy_flags():
    from inference_model_manager.dispatch import invoke_action
    from inference_models.models.auto_loaders.entities import PreProcessingOverrides

    model = _register_capturing_model()
    invoke_action(
        model,
        action="infer",
        images=["img"],
        confidence=0.5,
        disable_preproc_contrast=True,
        disable_preproc_static_crop=True,
    )

    assert model.received == {
        "confidence": 0.5,
        "pre_processing_overrides": PreProcessingOverrides(
            disable_contrast_enhancement=True,
            disable_grayscale=False,
            disable_static_crop=True,
        ),
    }


def test_invoke_action_keeps_explicit_overrides_when_flag_is_false():
    from inference_model_manager.dispatch import invoke_action
    from inference_models.models.auto_loaders.entities import PreProcessingOverrides

    model = _register_capturing_model()
    invoke_action(
        model,
        action="infer",
        images=["img"],
        pre_processing_overrides=PreProcessingOverrides(disable_static_crop=True),
        disable_preproc_contrast=False,
    )

    assert model.received == {
        "pre_processing_overrides": PreProcessingOverrides(
            disable_contrast_enhancement=False,
            disable_grayscale=False,
            disable_static_crop=True,
        ),
    }


def test_invoke_action_merges_true_flag_into_explicit_overrides():
    from inference_model_manager.dispatch import invoke_action
    from inference_models.models.auto_loaders.entities import PreProcessingOverrides

    model = _register_capturing_model()
    invoke_action(
        model,
        action="infer",
        images=["img"],
        pre_processing_overrides=PreProcessingOverrides(disable_static_crop=True),
        disable_preproc_contrast=True,
    )

    assert model.received == {
        "pre_processing_overrides": PreProcessingOverrides(
            disable_contrast_enhancement=True,
            disable_grayscale=False,
            disable_static_crop=True,
        ),
    }


def test_invoke_action_drops_legacy_flags_for_undeclared_entry():
    from inference_model_manager.dispatch import invoke_action

    model = _register_capturing_model(declare_flags=False)
    invoke_action(
        model,
        action="infer",
        images=["img"],
        confidence=0.5,
        disable_preproc_contrast=True,
        disable_preproc_static_crop=True,
    )

    assert model.received == {"confidence": 0.5}


def test_invoke_action_without_legacy_flags_passes_no_overrides():
    from inference_model_manager.dispatch import invoke_action

    model = _register_capturing_model()
    invoke_action(model, action="infer", images=["img"], confidence=0.5)

    assert model.received == {"confidence": 0.5}


def test_unpack_config_handles_7_and_8_tuples():
    from inference_model_manager.registry_defaults import _unpack_config

    seven = ("t", "m", True, {}, "v", "s", "r")
    assert _unpack_config(seven)[7] == {}
    eight = ("t", "m", True, {}, "v", "s", "r", {"a": "b"})
    assert _unpack_config(eight)[7] == {"a": "b"}
