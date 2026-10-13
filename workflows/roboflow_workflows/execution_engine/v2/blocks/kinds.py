"""Deprecated import path for ``roboflow_workflows.v2.data.image_kinds``.

Kept so existing V2 alpha callers keep working; it re-exports the same
objects. Import from the canonical module instead.
"""

from roboflow_workflows.v2.data.image_kinds import (
    CROP_SUMMARY_KIND,
    IMAGE_KIND,
    SERIALIZED_IMAGE_TYPE,
    V1_IMAGE_KEYS,
)

__all__ = [
    "CROP_SUMMARY_KIND",
    "IMAGE_KIND",
    "SERIALIZED_IMAGE_TYPE",
    "V1_IMAGE_KEYS",
]
