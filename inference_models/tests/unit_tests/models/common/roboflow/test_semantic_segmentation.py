import pytest
import torch

from inference_models.errors import CorruptedModelPackageError
from inference_models.models.common.roboflow.semantic_segmentation import (
    insert_background_class,
    resolve_background_class_id,
    validate_class_names,
)


def test_resolve_background_class_id_when_background_present() -> None:
    assert resolve_background_class_id(["background", "road", "sidewalk"]) == 0


def test_resolve_background_class_id_is_case_insensitive() -> None:
    assert resolve_background_class_id(["road", "Background", "sidewalk"]) == 1


def test_validate_class_names_with_minimal_binary_class_list() -> None:
    validate_class_names(["background", "object"])  # does not raise


def test_validate_class_names_with_multiclass() -> None:
    validate_class_names(["background", "road", "sidewalk"])  # does not raise


def test_validate_class_names_when_background_absent() -> None:
    with pytest.raises(CorruptedModelPackageError):
        validate_class_names(["road", "sidewalk", "building"])


def test_validate_class_names_when_no_foreground_class() -> None:
    with pytest.raises(CorruptedModelPackageError):
        validate_class_names(["background"])


def test_insert_background_class_when_background_first() -> None:
    out = insert_background_class(
        torch.tensor([0, 1, 2]), background_class_id=0, num_classes=4
    )
    assert out.tolist() == [1, 2, 3]


def test_insert_background_class_when_background_not_first() -> None:
    out = insert_background_class(
        torch.tensor([0, 1, 2]), background_class_id=1, num_classes=4
    )
    assert out.tolist() == [0, 2, 3]


def test_insert_background_class_binary_single_foreground() -> None:
    out = insert_background_class(
        torch.zeros(3, dtype=torch.long), background_class_id=0, num_classes=2
    )
    assert out.tolist() == [1, 1, 1]
