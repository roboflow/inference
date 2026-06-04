"""Semantic-segmentation helpers for Roboflow model packages.

These live apart from the generic `model_packages` parsing/config helpers
because they encode the semantic-segmentation *class-name contract* — a
`background` class plus at least one foreground class — shared by the
DeepLabV3+ and YOLO26-sem backends.
"""

from typing import List

from inference_models.errors import CorruptedModelPackageError


def validate_class_names(class_names: List[str]) -> None:
    """Validate the class names of a semantic-segmentation model package.

    A valid package declares a `background` class plus at least one foreground
    class, so `class_names` has >= 2 entries. `background` is required so that
    sub-threshold / unlabeled pixels map to a valid class id (a negative
    sentinel would alias a real class via negative indexing in downstream
    consumers — `class_names[-1]`, palette LUTs, the 0=background platform
    convention). At least one foreground class is required so consumers (e.g.
    the binary `nc==1` post-process) can assume a foreground class id exists.

    Raises `CorruptedModelPackageError` if either condition is unmet. Call this
    once at model load; downstream helpers (`resolve_background_class_id`, the
    post-process) then assume the precondition holds.
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
    """Return the index of the `background` class.

    Assumes the package has already been validated via `validate_class_names`
    (so `background` is present); call that at model load time.
    """
    return [c.lower() for c in class_names].index("background")
