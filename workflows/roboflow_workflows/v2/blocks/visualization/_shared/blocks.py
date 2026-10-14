"""Checks and drawing preparation shared by the batched detection painters.

The painter blocks live in ``native/<family>/v1.py``; each one checks its
batch with ``checked_batch`` and gets its drawings from ``checked_drawings``.

Every block takes whole batches (``batch="always"``), so the engine calls it
once per batch. Painters queue the batch's work on one CUDA stream and wait
once before returning (``streams.ReadyScope``). Images never leave the device.

Routes for boxes plus labels on ``B`` images::

    route                                            image   host       painter
                                                     copies  transfers  waits
    detection_visualization (fused, recommended)       B        1         1
    detection_drawing -> box -> label_in_place         B        1         2
    detection_drawing -> box -> label                  2B       1         2
    box -> label_in_place, both bound to Detections    B        2         2

A host transfer is one device-to-host copy of the batch's boxes, class ids
and confidences (``drawing.prepare_drawings``); the host waits for it. A
painter bound to native ``Detections`` makes it itself; bound to the output
of ``roboflow_core/detection_drawing@v1`` it makes none, so two painters share one.
A painter wait is the one host wait at the end of each painter call.
Drawing also waits for each image's box-table upload on CUDA (``kernels``);
those uploads are not included in the table's painter-wait count.

Mutation is declared, never implied. Copy painters return new images and
leave their inputs untouched. ``roboflow_core/simple_label_visualization_in_place@v1`` changes
its input image (``mutates``); bind it only to an image an earlier step of
this workflow created, such as the box painter's output. The compiler reports
conflicting readers of that image; it does not check who owns a passive
workflow input, so binding a caller's frame overwrites the caller's pixels.

Pixels are drawn by ``kernels.TensorKernels`` on CPU and CUDA. Workflow
definitions do not choose how pixels are drawn: a host that has other
implementations of these blocks adds them to its catalogue
(``Catalogue.with_contributions``).

Output images keep the input's identity, provenance, video metadata and
geometry (``ImageData.with_pixels``). Copies are contiguous and allocated on
the painter's stream; a consumer that queues asynchronous work on another
stream must ``record_stream`` them before dropping them.
"""

from typing import Any, List, Mapping, Sequence, Tuple
import numpy as np
import torch
from pydantic import ValidationError
from pydantic_core import PydanticCustomError
from roboflow_workflows.v2.blocks.visualization.drawing import (
    DetectionDrawing,
    prepare_drawings,
)
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.image import ImageData

_REMOVED_BACKEND_MESSAGE = (
    "no longer a painter parameter; delete it from the step. Painters draw "
    "with their built-in implementation. A host that provides other "
    "implementations adds them to its catalogue instead."
)


def reject_removed_backend(values: Any, *, params: str) -> Any:
    """Refuse a painter step that still sets the removed ``backend`` field.

    Without this, an old definition gets only "Extra inputs are not
    permitted", which does not say what replaced the field. The error is
    located at the field, like the other parameter errors.

    Args:
        values: Raw step parameters, before field validation.
        params: Name of the ``Params`` class, used as the error title.

    Returns:
        ``values``, unchanged.

    Raises:
        ValidationError: When ``values`` is a mapping with a ``backend`` key.
    """
    if isinstance(values, Mapping) and "backend" in values:
        removed = PydanticCustomError("removed_parameter", _REMOVED_BACKEND_MESSAGE)
        raise ValidationError.from_exception_data(
            params,
            [{"type": removed, "loc": ("backend",), "input": values["backend"]}],
        )

    return values


def checked_batch(
    image: Batch, predictions: Batch, *, block: str
) -> Tuple[List[ImageData], List[Any], torch.device]:
    """Check a painter's batch before anything is prepared or painted.

    Args:
        image: The batch's images.
        predictions: Their detections or prepared drawings, same order.
        block: Type of the calling block, named in errors.

    Returns:
        The images, the predictions and the one device of the images (CPU
        for an empty batch).

    Raises:
        ValueError: On misaligned batches, a non-RGB image or images on
            several devices.
    """
    images, members = list(image), list(predictions)
    if len(images) != len(members):
        raise ValueError(
            f"{block}: image and predictions batches must align: "
            f"{len(images)} != {len(members)}"
        )
    for member in images:
        if member.channels != 3:
            raise ValueError(f"{block} needs 3-channel RGB images, got {member!r}")

    devices = {member.tensor_image.device for member in images}
    if len(devices) > 1:
        raise ValueError(
            f"{block}: a batch's images must be on one device, got "
            f"{sorted(map(str, devices))}"
        )
    device = devices.pop() if devices else torch.device("cpu")

    return images, members, device


def checked_drawings(members: Sequence[Any]) -> List[DetectionDrawing]:
    """Return one drawing per member, preparing native detections together.

    Native ``Detections`` are prepared with one host transfer; prepared
    drawings are used as they are.

    Args:
        members: Detections or prepared drawings, in batch order.

    Returns:
        The drawings, in the same order.

    Raises:
        ValueError: On a class id without a name or a negative class id.
    """
    native = [
        position
        for position, member in enumerate(members)
        if not isinstance(member, DetectionDrawing)
    ]
    drawings = list(members)
    prepared = prepare_drawings([members[position] for position in native])
    for position, drawing in zip(native, prepared):
        drawings[position] = drawing

    for drawing in drawings:
        if (np.asarray(drawing.class_id) < 0).any():
            raise ValueError("cannot color a detection with a negative class id")

    return drawings
