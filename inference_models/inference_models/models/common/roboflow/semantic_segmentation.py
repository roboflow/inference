"""Semantic-segmentation class-name helpers shared by the DeepLabV3+ and
YOLO26-sem backends (kept out of the generic `model_packages` module)."""

from typing import List

import torch

from inference_models.errors import CorruptedModelPackageError


def validate_class_names(class_names: List[str]) -> None:
    """Require a `background` class plus >= 1 foreground class, raising otherwise.

    Call once at model load so downstream helpers can assume the precondition.
    """
    if "background" not in [c.lower() for c in class_names]:
        raise CorruptedModelPackageError(
            message="Semantic segmentation model package does not define a `background` class in "
            "`class_names.txt`. A background class is required so that sub-threshold pixels map to a "
            "valid class id. If you created the model package manually, prepend `background` to the class "
            "names. If the weights are hosted on the Roboflow platform - contact support.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )
    if len(class_names) < 2:
        raise CorruptedModelPackageError(
            message="Semantic segmentation model package must define `background` plus at least one "
            f"foreground class, but `class_names.txt` only contains {class_names}. If you created the "
            "model package manually, ensure it lists `background` followed by the foreground class(es). "
            "If the weights are hosted on the Roboflow platform - contact support.",
            help_url="https://inference-models.roboflow.com/errors/model-loading/#corruptedmodelpackageerror",
        )


def resolve_background_class_id(class_names: List[str]) -> int:
    """Index of the `background` class; assumes a `validate_class_names`-checked package."""
    return [c.lower() for c in class_names].index("background")


def insert_background_class(
    class_ids: torch.Tensor, *, background_class_id: int, num_classes: int
) -> torch.Tensor:
    """Map foreground-channel indices (``0..K-1``) to full class ids, skipping the
    background slot. ``num_classes`` is the full class count (``K + 1``)."""
    foreground_ids = torch.tensor(
        [i for i in range(num_classes) if i != background_class_id],
        device=class_ids.device,
        dtype=class_ids.dtype,
    )
    return foreground_ids[class_ids]
