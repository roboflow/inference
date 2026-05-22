"""
Tensor-native sibling of `common/serializers.py`. Per the plan's locked
decision [ITERATE 3.B], we ship this file from day one so the loader
can swap imports symmetrically with the deserializer side.

The lazy `tensor_image -> numpy_image` fallback inside
`WorkflowImageData` (via `base64_image -> numpy_image`) means the numpy
`serialise_image` already produces correct, JSON-byte-identical output
when called against a tensor-backed `WorkflowImageData`. So this module
currently re-exports the numpy implementations. Future optimisations
(e.g., a tensor -> JPEG path that bypasses the numpy cache) can replace
the bound names without changing the loader-facing contract.
"""

from inference.core.workflows.core_steps.common.serializers import (
    serialise_image,
    serialise_rle_sv_detections,
    serialise_sv_detections,
)

__all__ = [
    "serialise_image",
    "serialise_sv_detections",
    "serialise_rle_sv_detections",
]
