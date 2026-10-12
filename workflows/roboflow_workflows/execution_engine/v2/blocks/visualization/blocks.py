"""Batched detection painters and their optional shared drawing preparation.

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
of ``v2/detection_drawing@v1`` it makes none, so two painters share one.
A painter wait is the one host wait at the end of each painter call. The
tensor backend also waits for each image's box-table upload on CUDA; those
uploads are not included in the table's painter-wait count.

Mutation is declared, never implied. Copy painters return new images and
leave their inputs untouched. ``v2/label_visualization_in_place@v1`` changes
its input image (``mutates``); bind it only to an image an earlier step of
this workflow created, such as the box painter's output. The compiler reports
conflicting readers of that image; it does not check who owns a passive
workflow input, so binding a caller's frame overwrites the caller's pixels.

``backend`` selects how pixels are drawn (``kernels``): ``"tensor"`` (CPU and
CUDA) or ``"triton"`` (CUDA with Triton, one launch per call; imported on
first use). Both paint the same pixels; there is no fallback between them.

Output images keep the input's identity, provenance, video metadata and
geometry (``ImageData.with_pixels``). Copies are contiguous and allocated on
the painter's stream; a consumer that queues asynchronous work on another
stream must ``record_stream`` them before dropping them.
"""

from typing import Any, Dict, List, Literal, Sequence, Tuple

import numpy as np
import torch
from pydantic import Field
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.drawing import (
    DETECTION_DRAWING_KIND,
    DetectionDrawing,
    prepare_drawings,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.kernels import (
    DrawingKernels,
    create_kernels,
)
from roboflow_workflows.execution_engine.v2.blocks.visualization.streams import (
    ReadyScope,
)
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)

DrawingBackend = Literal["tensor", "triton"]

BACKEND_DESCRIPTION = (
    "How pixels are drawn: 'tensor' (V1 tensor primitives, CPU and CUDA) or "
    "'triton' (one Triton launch per batch, CUDA only, needs the triton "
    "package). Both draw the same pixels; there is no fallback."
)


class DetectionDrawingBlock(Block):
    """Prepare a batch's detections for drawing with one host transfer.

    Optional. Bind its ``drawing`` output to several painters so they share
    one device-to-host transfer instead of making one each.
    """

    type = "v2/detection_drawing@v1"
    outputs = {
        "drawing": Output(
            DETECTION_DRAWING_KIND,
            source="predictions",
            description="New read-only host copy of boxes, classes and label texts.",
        ),
    }

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        predictions: Ref(
            OBJECT_DETECTION_PREDICTION_KIND, batch="always"  # noqa: F821
        ) = Field(
            description=(
                "Detections of every image, delivered as one batch; each needs "
                "image_metadata['class_names'] unless empty."
            )
        )

    def __init__(self) -> None:
        self._scope = ReadyScope()

    def run(self, *, predictions: Batch) -> List[Dict[str, Any]]:
        """Copy every image's detections to the host in one transfer.

        Args:
            predictions: Detections of the batch's images; not modified.

        Returns:
            One ``drawing`` per image, in batch order.

        Raises:
            ValueError: On tensors on several devices, inconsistent shapes or
                a class id without a name.
        """
        members = list(predictions)
        if not members:
            return []

        # Keep the device copies off the caller's stream after joining its
        # queued work. prepare_drawings itself waits for the host transfer.
        with self._scope.work(members[0].xyxy.device):
            drawings = prepare_drawings(members)
        results = [{"drawing": drawing} for drawing in drawings]

        return results


class _PainterParams(BlockParams):
    # noqa F821: flake8 parses the string "always" in an annotation as a name.
    image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
        description="RGB images to annotate, delivered as one batch; not modified."
    )
    predictions: Ref(
        OBJECT_DETECTION_PREDICTION_KIND,
        DETECTION_DRAWING_KIND,
        batch="always",  # noqa: F821
    ) = Field(
        description=(
            "Per image, in batch order: native Detections (with "
            "image_metadata['class_names']) or a prepared DetectionDrawing."
        )
    )
    backend: DrawingBackend = Field(
        default="tensor",
        description=BACKEND_DESCRIPTION,
        examples=["tensor", "triton"],
    )


class _Painter(Block):
    """Shared body of the painters: checks, one stream scope, one wait."""

    def __init__(self) -> None:
        self._scope = ReadyScope()
        # One kernel set per backend name; ``backend`` is a literal parameter,
        # so a step uses one. Each holds bounded caches (see ``kernels``).
        self._kernels: Dict[str, DrawingKernels] = {}

    def _kernels_for(self, backend: str) -> DrawingKernels:
        if backend not in self._kernels:
            self._kernels[backend] = create_kernels(backend)

        return self._kernels[backend]

    def _paint_copies(
        self,
        image: Batch,
        predictions: Batch,
        *,
        backend: str,
        boxes: bool,
        labels: bool,
    ) -> List[Dict[str, Any]]:
        images, members, device = _checked_batch(image, predictions, block=self.type)
        if not images:
            return []

        kernels = self._kernels_for(backend)
        with self._scope.work(device):
            drawings = _drawings(members)
            scenes = [
                member.tensor_image.clone(memory_format=torch.contiguous_format)
                for member in images
            ]
            kernels.paint(scenes, drawings, boxes=boxes, labels=labels)
        results = [
            {"image": member.with_pixels(scene)}
            for member, scene in zip(images, scenes)
        ]

        return results


class BoxVisualizationBlock(_Painter):
    """Draw each detection's box border onto a copy of its image.

    V1 default style: supervision default palette by class id, 2 px square
    borders. One wait per batch.
    """

    type = "v2/box_visualization@v1"
    outputs = {
        "image": Output(IMAGE_KIND, description="New image with boxes."),
    }
    Params = _PainterParams

    def run(
        self, *, image: Batch, predictions: Batch, backend: str = "tensor"
    ) -> List[Dict[str, Any]]:
        """Copy every image and draw its boxes, then wait once.

        Args:
            image: The batch's RGB images, all on one device; not modified.
            predictions: Their detections or prepared drawings, same order.
            backend: ``"tensor"`` or ``"triton"``.

        Returns:
            Per image, in order: ``image``, a new image with the source's
            identity and provenance, ready to read.

        Raises:
            ValueError: Before any painting, on misaligned batches, a non-RGB
                image, images on several devices, a class id without a name
                or a negative class id.
            DrawingBackendUnavailable: ``triton`` without CUDA or Triton.
        """
        results = self._paint_copies(
            image, predictions, backend=backend, boxes=True, labels=False
        )

        return results


class LabelVisualizationBlock(_Painter):
    """Paste each detection's ``"<class> <confidence>"`` label onto a copy.

    V1 default style: white text on a patch of the box color, above the box's
    top-left corner. One wait per batch. To label the box painter's output
    without a second copy, use ``v2/label_visualization_in_place@v1``.
    """

    type = "v2/label_visualization@v1"
    outputs = {
        "image": Output(IMAGE_KIND, description="New image with labels."),
    }
    Params = _PainterParams

    def run(
        self, *, image: Batch, predictions: Batch, backend: str = "tensor"
    ) -> List[Dict[str, Any]]:
        """Copy every image and paste its labels, then wait once.

        Args:
            image: The batch's RGB images, all on one device; not modified.
            predictions: Their detections or prepared drawings, same order.
            backend: ``"tensor"`` or ``"triton"``.

        Returns:
            Per image, in order: ``image``, a new labelled image with the
            source's identity and provenance, ready to read.

        Raises:
            ValueError: Before any painting, as ``v2/box_visualization@v1``.
            DrawingBackendUnavailable: ``triton`` without CUDA or Triton.
        """
        results = self._paint_copies(
            image, predictions, backend=backend, boxes=False, labels=True
        )

        return results


class DetectionVisualizationBlock(_Painter):
    """Draw boxes, then labels, onto one copy per image: the fast route.

    Same pixels as the box painter followed by the label painter, with one
    copy, at most one host transfer and one wait per batch.
    """

    type = "v2/detection_visualization@v1"
    outputs = {
        "image": Output(IMAGE_KIND, description="New image with boxes and labels."),
    }
    Params = _PainterParams

    def run(
        self, *, image: Batch, predictions: Batch, backend: str = "tensor"
    ) -> List[Dict[str, Any]]:
        """Copy every image, draw its boxes and labels, then wait once.

        Args:
            image: The batch's RGB images, all on one device; not modified.
            predictions: Their detections or prepared drawings, same order.
            backend: ``"tensor"`` or ``"triton"``.

        Returns:
            Per image, in order: ``image``, a new annotated image with the
            source's identity and provenance, ready to read.

        Raises:
            ValueError: Before any painting, as ``v2/box_visualization@v1``.
            DrawingBackendUnavailable: ``triton`` without CUDA or Triton.
        """
        results = self._paint_copies(
            image, predictions, backend=backend, boxes=True, labels=True
        )

        return results


class _InPlaceParams(_PainterParams):
    # noqa F821: flake8 parses the string "always" in an annotation as a name.
    image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
        description=(
            "Contiguous RGB images this workflow created (e.g. the box "
            "painter's output), changed in place; delivered as one batch."
        )
    )


class InPlaceLabelVisualizationBlock(_Painter):
    """Paste labels in place on images an earlier step of this workflow created.

    Declares ``mutates=("image",)``. Meant to follow the box painter, so boxes
    plus labels cost one copy without the fused block. Bound to a passive
    workflow input it overwrites the caller's pixels; the compiler cannot see
    that ownership, so do not do it.

    On failure the batch's scope is still waited for. With the tensor
    backend, images before a failing one stay labelled; triton checks the
    whole batch before its launch.
    """

    type = "v2/label_visualization_in_place@v1"
    mutates = ("image",)
    outputs = {
        "image": Output(
            IMAGE_KIND, source="image", description="The same image, now labelled."
        ),
    }
    Params = _InPlaceParams

    def run(
        self, *, image: Batch, predictions: Batch, backend: str = "tensor"
    ) -> List[Dict[str, Any]]:
        """Paste every image's labels in place, then wait once.

        Args:
            image: Contiguous RGB images on one device; changed in place.
            predictions: Their detections or prepared drawings, same order.
            backend: ``"tensor"`` or ``"triton"``.

        Returns:
            Per image, in order: ``image``, the same object, ready to read.

        Raises:
            ValueError: Before any painting, as ``v2/box_visualization@v1``,
                or on a non-contiguous image.
            DrawingBackendUnavailable: ``triton`` without CUDA or Triton.
        """
        images, members, device = _checked_batch(image, predictions, block=self.type)
        for member in images:
            if not member.tensor_image.is_contiguous():
                raise ValueError(
                    f"{self.type} paints in place and needs contiguous image "
                    "tensors; bind the box painter's output"
                )
        if not images:
            return []

        kernels = self._kernels_for(backend)
        with self._scope.work(device):
            drawings = _drawings(members)
            scenes = [member.tensor_image for member in images]
            kernels.paint(scenes, drawings, boxes=False, labels=True)
        results = [{"image": member} for member in images]

        return results


def _checked_batch(
    image: Batch, predictions: Batch, *, block: str
) -> Tuple[List[ImageData], List[Any], torch.device]:
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


def _drawings(members: Sequence[Any]) -> List[DetectionDrawing]:
    # Native Detections are prepared together (one transfer); prepared
    # drawings are used as they are.
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


BLOCKS = (
    DetectionDrawingBlock,
    BoxVisualizationBlock,
    LabelVisualizationBlock,
    DetectionVisualizationBlock,
    InPlaceLabelVisualizationBlock,
)
