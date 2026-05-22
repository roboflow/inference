"""
Tensor-native sibling of `common/deserializers.py`. Same public function
names; loader swaps the import based on `ENABLE_TENSOR_DATA_REPRESENTATION`.

Per the plan's locked decision [ITERATE 4.A], the numpy file is left
untouched. Functions here add tensor-aware code paths and delegate to
the numpy implementations for everything else.
"""

from typing import Any

import torch

from inference.core.workflows.core_steps.common.deserializers import (
    _parse_optional_parent_metadata,
    deserialize_detections_kind,
    deserialize_image_kind as _deserialize_image_kind_numpy,
    deserialize_rle_detections_kind,
    deserialize_video_metadata_kind,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.constants import (
    PARENT_ID_KEY,
    PARENT_ORIGIN_KEY,
    ROOT_PARENT_ID_KEY,
    ROOT_PARENT_ORIGIN_KEY,
)
from inference.core.workflows.execution_engine.entities.base import (
    WorkflowImageData,
)


def deserialize_image_kind(
    parameter: str,
    image: Any,
    prevent_local_images_loading: bool = False,
) -> WorkflowImageData:
    if isinstance(image, WorkflowImageData):
        return image
    if isinstance(image, torch.Tensor):
        parent_metadata, workflow_root_ancestor_metadata, video_metadata = (
            _parse_image_metadata_fields(parameter=parameter, image=None)
        )
        return WorkflowImageData(
            parent_metadata=parent_metadata,
            workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
            tensor_image=image,
            video_metadata=video_metadata,
        )
    if isinstance(image, dict) and image.get("type") == "tensor":
        value = image.get("value")
        if not isinstance(value, torch.Tensor):
            raise RuntimeInputError(
                public_message=(
                    f"Detected runtime parameter `{parameter}` declared with "
                    f"type='tensor' but its value is of type {type(value)}; "
                    "expected torch.Tensor."
                ),
                context="workflow_execution | runtime_input_validation",
            )
        parent_metadata, workflow_root_ancestor_metadata, video_metadata = (
            _parse_image_metadata_fields(parameter=parameter, image=image)
        )
        return WorkflowImageData(
            parent_metadata=parent_metadata,
            workflow_root_ancestor_metadata=workflow_root_ancestor_metadata,
            tensor_image=value,
            video_metadata=video_metadata,
        )
    return _deserialize_image_kind_numpy(
        parameter=parameter,
        image=image,
        prevent_local_images_loading=prevent_local_images_loading,
    )


def _parse_image_metadata_fields(parameter: str, image: Any):
    """Parse the parent/root-parent/video metadata block shared by the
    image-kind deserializer paths. Mirrors the prefix of
    `deserialize_image_kind` in the numpy file."""
    is_image_dict = isinstance(image, dict)
    parent_id = image.get(PARENT_ID_KEY, parameter) if is_image_dict else parameter
    parent_origin = image.get(PARENT_ORIGIN_KEY) if is_image_dict else None
    parent_metadata = _parse_optional_parent_metadata(
        parameter=parameter,
        parent_id=parent_id,
        parent_origin=parent_origin,
    )
    root_parent_id = image.get(ROOT_PARENT_ID_KEY) if is_image_dict else None
    root_parent_origin = (
        image.get(ROOT_PARENT_ORIGIN_KEY) if is_image_dict else None
    )
    workflow_root_ancestor_metadata = _parse_optional_parent_metadata(
        parameter=parameter,
        parent_id=root_parent_id,
        parent_origin=root_parent_origin,
    )
    video_metadata = None
    if is_image_dict and "video_metadata" in image:
        video_metadata = deserialize_video_metadata_kind(
            parameter=parameter, video_metadata=image["video_metadata"]
        )
    return parent_metadata, workflow_root_ancestor_metadata, video_metadata


__all__ = [
    "deserialize_image_kind",
    "deserialize_detections_kind",
    "deserialize_rle_detections_kind",
]
