"""Unit tests for dispatch module — task resolution and listing."""

from inference_model_manager.dispatch import list_tasks_by_mro_names
from inference_model_manager.registry_defaults import lazy_register_by_names


def test_list_tasks_by_mro_names_object_detection():
    """ObjectDetectionModel MRO names should return 'infer' task."""
    lazy_register_by_names(["ObjectDetectionModel"])
    tasks = list_tasks_by_mro_names(["ObjectDetectionModel"])
    assert "infer" in tasks
    assert tasks["infer"]["default"] is True
    assert "images" in tasks["infer"]["params"]


def test_list_tasks_by_mro_names_unknown_class():
    """Unknown class names should return empty dict."""
    tasks = list_tasks_by_mro_names(["CompletelyUnknownModel"])
    assert tasks == {}


def test_list_tasks_by_mro_names_walks_mro():
    """Should match on any ancestor in the MRO list."""
    lazy_register_by_names(["ObjectDetectionModel"])
    tasks = list_tasks_by_mro_names(
        [
            "YOLOv8ForObjectDetectionTorchScript",
            "ObjectDetectionModel",
            "object",
        ]
    )
    assert "infer" in tasks


def test_invoke_task_applies_param_aliases():
    from inference_model_manager.dispatch import invoke_task
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
    result = invoke_task(AliasedModel(), task="query", prompt="hi")
    assert result == "answer:hi"


def test_unpack_config_handles_7_and_8_tuples():
    from inference_model_manager.registry_defaults import _unpack_config

    seven = ("t", "m", True, {}, "v", "s", "r")
    assert _unpack_config(seven)[7] == {}
    eight = ("t", "m", True, {}, "v", "s", "r", {"a": "b"})
    assert _unpack_config(eight)[7] == {"a": "b"}
