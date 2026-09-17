"""Tensor-data-representation sibling of ``fusion/frame_delay/v1``.

The delay machinery (per-video ring buffer keyed by frame number, LRU stream
tracking, frame-number-restart detection, window eviction) is
representation-agnostic and reused verbatim from the numpy sibling by
subclassing it. The ONLY behavioral change is at buffer-insertion time: image
payloads are spilled to host memory before they are buffered.

Why: under ``ENABLE_TENSOR_DATA_REPRESENTATION`` a ``WorkflowImageData`` may
carry its pixels as a CHW uint8 CUDA tensor (``tensor_image``). Video frames
delivered by the Jetson tensor bridge borrow buffers from a small (8-slot)
per-pipeline CUDA pool that are recycled only once the tensor reference is
dropped. Buffering such frames verbatim pins ~11 MB of device memory per 2K
frame for the whole delay window, and ``|offset| > 8`` starves the bridge pool
outright, degrading every subsequent frame to fresh ``cudaMalloc`` churn.
Spilling at insertion converts that into ~11 MB of ordinary host memory per
buffered 2K frame and returns the pool buffer immediately.

What is spilled: every ``WorkflowImageData`` found at the top level of ``data``
or nested inside list/tuple containers - mirroring what the numpy manifest can
actually be wired with (``IMAGE_KIND`` at the top level; ``LIST_OF_VALUES_KIND``
e.g. a collapsed list of crops). Each spilled entry is a NEW
``WorkflowImageData`` built via ``WorkflowImageData.copy_and_replace(...,
numpy_image=...)``: reading ``numpy_image`` first materialises host pixels
(the D2H conversion inside ``WorkflowImageData.numpy_image`` copies, so the
array does not alias the tensor storage), and ``copy_and_replace`` called with
an image-representation kwarg rebuilds the instance with every non-passed
representation slot set to ``None`` - the result provably holds no
``_tensor_image``, while ``parent_metadata``, ``workflow_root_ancestor_metadata``
and ``video_metadata`` are carried over as-is. Images whose tensor was never
materialised (``is_tensor_materialised()`` is ``False``) are buffered
untouched: they hold no device memory, and spilling them would force an eager
decode of base64/reference-born images. On emission the stored host-resident
object is returned directly; downstream tensor-mode blocks re-materialise
``tensor_image`` lazily on first access (re-upload to
``WORKFLOWS_IMAGE_TENSOR_DEVICE``).

Deliberately NOT spilled: every non-image payload - predictions, numbers,
strings, dicts, arbitrary objects - is buffered untouched, INCLUDING
tensor-native ``Detections`` / ``InstanceDetections`` whose boxes / masks live
on GPU. Delaying mask-carrying predictions therefore still retains their GPU
memory for the delay window; the image spill is the load-bearing fix, because
full frames dominate the footprint and pin the bridge buffer pool. Images
nested deeper than list/tuple containers (e.g. inside dicts) are likewise
buffered as-is.
"""

from typing import Any, Optional, Type, Union

from pydantic import ConfigDict

from inference.core.workflows.core_steps.fusion.frame_delay.v1 import (
    LONG_DESCRIPTION as NUMPY_LONG_DESCRIPTION,
)
from inference.core.workflows.core_steps.fusion.frame_delay.v1 import (
    BlockManifest as NumpyBlockManifest,
)
from inference.core.workflows.core_steps.fusion.frame_delay.v1 import (
    FrameDelayBlockV1 as NumpyFrameDelayBlockV1,
)
from inference.core.workflows.execution_engine.entities.base import WorkflowImageData
from inference.core.workflows.prototypes.block import BlockResult, WorkflowBlockManifest

TENSOR_MODE_ADDENDUM = """
## Tensor Data Representation Behavior

Under `ENABLE_TENSOR_DATA_REPRESENTATION`, image payloads wired into `data` may
carry their pixels as a CUDA tensor; on Jetson video pipelines that tensor is a
buffer borrowed from a small per-pipeline CUDA pool which is recycled only once
the tensor reference is dropped. This variant therefore spills image payloads
to host memory at buffering time: the buffered (and later emitted) image holds
host-resident `numpy_image` pixels and no tensor reference, so device memory is
released immediately instead of being pinned for the delay window. The cost is
~11 MB of host memory per buffered 2K frame (~6 MB at 1080p, ~25 MB at 4K);
downstream consumers of the delayed image re-upload it to the device lazily on
first `tensor_image` access. Non-image payloads (predictions, numbers, dicts,
...) are buffered untouched - delaying tensor-native predictions that carry GPU
masks still retains their GPU memory for the delay window.
"""

LONG_DESCRIPTION = NUMPY_LONG_DESCRIPTION + TENSOR_MODE_ADDENDUM


class BlockManifest(NumpyBlockManifest):
    model_config = ConfigDict(
        json_schema_extra={
            **NumpyBlockManifest.model_config["json_schema_extra"],
            "long_description": LONG_DESCRIPTION,
        }
    )


def _spill_images_to_host(data: Any) -> Any:
    """Returns `data` with every `WorkflowImageData` found at the top level or
    inside list/tuple containers replaced by a host-resident copy that holds no
    tensor reference (see the module docstring). Containers are rebuilt only
    when an element actually changed; everything else - non-image payloads,
    images with no materialised tensor - is returned untouched, preserving
    object identity exactly like the numpy sibling does."""
    if isinstance(data, WorkflowImageData):
        if not data.is_tensor_materialised():
            # Already host-resident (or a base64/reference-born image that
            # never materialised a tensor): buffering it holds no device
            # memory, and spilling would force an eager decode.
            return data
        # Reading `numpy_image` materialises host pixels (the D2H conversion
        # copies, so the array does not alias tensor storage); passing it as an
        # image-representation kwarg makes `copy_and_replace` null every other
        # representation slot, so the copy provably drops the tensor reference
        # while keeping parent / root-ancestor / video metadata intact.
        return WorkflowImageData.copy_and_replace(
            origin_image_data=data, numpy_image=data.numpy_image
        )
    if isinstance(data, (list, tuple)):
        spilled = [_spill_images_to_host(data=element) for element in data]
        if all(new is original for new, original in zip(spilled, data)):
            return data
        return tuple(spilled) if isinstance(data, tuple) else spilled
    return data


class FrameDelayBlockV1(NumpyFrameDelayBlockV1):
    """Numpy frame-delay block with image payloads spilled to host memory at
    buffer-insertion time - see the module docstring for the full policy."""

    @classmethod
    def get_manifest(cls) -> Type[WorkflowBlockManifest]:
        return BlockManifest

    def run(
        self,
        image: WorkflowImageData,
        data: Any,
        offset: int,
        default_value: Optional[Union[bool, int, float, str]] = None,
    ) -> BlockResult:
        return super().run(
            image=image,
            data=_spill_images_to_host(data=data),
            offset=offset,
            default_value=default_value,
        )
