import pytest

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.roboflow.semantic_segmentation import (
    resolve_background_class_id,
    validate_class_names,
)


def test_resolve_background_class_id_when_background_present() -> None:
    assert resolve_background_class_id(["background", "road", "sidewalk"]) == 0


def test_resolve_background_class_id_is_case_insensitive() -> None:
    assert resolve_background_class_id(["road", "Background", "sidewalk"]) == 1


def test_validate_class_names_with_minimal_binary_class_list() -> None:
    # background + a single foreground class is the minimal valid package
    validate_class_names(["background", "object"])  # does not raise


def test_validate_class_names_with_multiclass() -> None:
    validate_class_names(["background", "road", "sidewalk"])  # does not raise


def test_validate_class_names_when_background_absent() -> None:
    with pytest.raises(CorruptedModelPackageError):
        validate_class_names(["road", "sidewalk", "building"])


def test_validate_class_names_when_no_foreground_class() -> None:
    # background present but no foreground class -> invalid semantic-seg package
    with pytest.raises(CorruptedModelPackageError):
        validate_class_names(["background"])
