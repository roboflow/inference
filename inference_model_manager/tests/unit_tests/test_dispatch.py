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
