"""Shared detection drawing preparation and painters that consume it.

Three ways to draw the same boxes and labels, all with separate box and label
steps::

    per-painter (10)     predictions ─┬─ boxes  (2 device-to-host copies of predictions)
                                      └─ labels (4 device-to-host copies of predictions)

    shared-prep          predictions ── prep ── drawings ─┬─ boxes  (N calls, host data only)
                         (N images)     1 transfer/batch  └─ labels (N calls, host data only)

    batch-painters       predictions ── prep ── drawings ─┬─ batch boxes  (1 call)
                         (N images)     1 transfer/batch  └─ batch labels (1 call)

Both shared variants use the same prep step (``BatchedDetectionDrawingPrep``,
one engine call per run). It copies every image's ``xyxy``, ``class_id`` and
``confidence`` to the host and formats the label texts. The work per run::

    device: view each tensor as bytes and concatenate them (small copies;
            strided model columns are gathered first)
    one device-to-host transfer of those bytes
    host:   copy into an immutable ``bytes`` object, view it as the three arrays

Bytes, not a common dtype, so nothing is rounded: an int64 class id above
2**24 or 2**53 arrives as it left. The result is a ``DetectionDrawing``, a new
immutable value per call: its arrays are read-only views of a ``bytes``
object, so neither a painter nor a later change of the source tensors can
alter it. Nothing is cached on ``Detections`` or across calls. The
``predictions`` output itself stays the detector's native ``Detections`` on
its device.

The painters draw exactly what the 09 painters draw from the same numbers: the
same 08 palette, ``gpu_draw_boxes``, label placement, sprite cache, pinned
paste ring and readiness. Every block here returns ready values: device work
runs on the block's own stream, and the block waits for an event recorded
after it (09 ``_ReadyOnReturn``). Nothing calls ``torch.cuda.synchronize()``.

The batch painters take ``batch="always"`` inputs, so the engine calls each
of them once per run. Inside, they call the per-image painter once per image,
with the same primitives and the same readiness wait per image. So
``shared-prep`` and ``batch-painters`` differ only in painter delivery: how
many engine calls the painters take and in which order the engine schedules
them. Prep, transfers and per-image events are the same.
"""

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import structural_imports  # noqa: F401 - installs the 08, 09 and 10 search paths
import torch
from pydantic import Field
from roboflow_workflows.core_steps.visualizations.bounding_box.v1_tensor import (
    gpu_draw_boxes,
)
from roboflow_workflows.core_steps.visualizations.label.v1_tensor import (
    _LabelSprite,
    gpu_paste_label_sprites,
)
from roboflow_workflows.execution_engine.v2.blocks.image_data import ImageData
from roboflow_workflows.execution_engine.v2.blocks.kinds import IMAGE_KIND
from roboflow_workflows.execution_engine.v2.blocks.predictions import (
    OBJECT_DETECTION_PREDICTION_KIND,
)
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.data import Batch
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, Kind

# isort: split

import batched_blocks
from detection_blocks import BOX_THICKNESS, PALETTE, _elapsed_ms, _require_rgb
from gpu_blocks import GpuBoxVisualization, GpuLabelVisualization, _ReadyOnReturn

from inference_models.models.base.object_detection import Detections


@dataclass(frozen=True)
class DetectionDrawing:
    """What the painters need of one image's detections, on the host.

    Built per call by ``prepare_drawings``; never cached or shared between
    calls. Arrays are read-only views of an immutable ``bytes`` object.

    Args:
        xyxy: ``(N, 4)`` boxes, the dtype of ``Detections.xyxy``.
        class_id: ``(N,)`` class ids, the dtype of ``Detections.class_id``.
        confidence: ``(N,)`` confidences, the dtype of ``Detections.confidence``.
        labels: ``N`` texts ``"<class name> <confidence:.2f>"``.

    Raises:
        ValueError: On mismatched lengths, wrong ranks or writeable arrays.
    """

    xyxy: np.ndarray
    class_id: np.ndarray
    confidence: np.ndarray
    labels: Tuple[str, ...]

    def __post_init__(self) -> None:
        count = len(self.labels)
        shapes_ok = (
            self.xyxy.shape == (count, 4)
            and self.class_id.shape == (count,)
            and self.confidence.shape == (count,)
        )
        if not shapes_ok:
            raise ValueError(
                f"DetectionDrawing of {count} labels has xyxy {self.xyxy.shape}, "
                f"class_id {self.class_id.shape}, confidence {self.confidence.shape}"
            )
        if any(array.flags.writeable for array in self._arrays()):
            raise ValueError("DetectionDrawing arrays must be read-only")

    def __len__(self) -> int:
        return len(self.labels)

    def _arrays(self) -> Tuple[np.ndarray, ...]:
        return self.xyxy, self.class_id, self.confidence


def _is_detection_drawing(payload: Any) -> bool:
    return isinstance(payload, DetectionDrawing)


DETECTION_DRAWING_KIND = Kind(
    name="detection_drawing",
    description=(
        "Experiment 11 DetectionDrawing: one image's boxes, class ids, "
        "confidences and label texts as read-only host arrays."
    ),
    validate=_is_detection_drawing,
)


def prepare_drawings(predictions: Sequence[Detections]) -> List[DetectionDrawing]:
    """Copy the drawing data of every ``Detections`` to the host at once.

    One device-to-host transfer for the whole sequence. On CUDA the caller
    runs this inside its stream context; the transfer is complete on return.

    Args:
        predictions: Native detections, all tensors on one device. Each needs
            ``image_metadata["class_names"]`` naming its class ids, unless it
            is empty.

    Returns:
        One new ``DetectionDrawing`` per element, in order.

    Raises:
        ValueError: On tensors on different devices, inconsistent shapes or
            a class id without a name.
    """
    tensors = []
    for detections in predictions:
        tensors.extend(_drawing_tensors(detections))
    _require_one_device(tensors)

    host = _to_host(tensors)
    drawings = []
    for position, detections in enumerate(predictions):
        xyxy, class_id, confidence = host[3 * position : 3 * position + 3]
        labels = _class_and_confidence_labels(
            class_id, confidence, image_metadata=detections.image_metadata
        )
        drawings.append(
            DetectionDrawing(
                xyxy=xyxy, class_id=class_id, confidence=confidence, labels=labels
            )
        )

    return drawings


def _drawing_tensors(detections: Detections) -> Tuple[torch.Tensor, ...]:
    count = len(detections)
    xyxy, class_id, confidence = (
        detections.xyxy,
        detections.class_id,
        detections.confidence,
    )
    if xyxy.shape != (count, 4) or class_id.shape != (count,):
        raise ValueError(
            f"Detections xyxy {tuple(xyxy.shape)} and class_id "
            f"{tuple(class_id.shape)} do not describe {count} boxes"
        )
    if confidence.shape != (count,):
        raise ValueError(
            f"Detections confidence {tuple(confidence.shape)} does not describe "
            f"{count} boxes"
        )

    return xyxy, class_id, confidence


def _require_one_device(tensors: Sequence[torch.Tensor]) -> None:
    devices = {tensor.device for tensor in tensors}
    if len(devices) > 1:
        raise ValueError(
            f"detections to draw must be on one device, got {sorted(map(str, devices))}"
        )


def _to_host(tensors: Sequence[torch.Tensor]) -> List[np.ndarray]:
    # Raw bytes in one buffer: small device copies (strided columns are
    # gathered, then concatenated), one device-to-host transfer, one host
    # copy into immutable bytes. Exact for every dtype.
    if not tensors:
        return []

    raw = [_dense(tensor).view(torch.uint8) for tensor in tensors]
    blob = torch.cat(raw).cpu().numpy().tobytes()

    arrays = []
    offset = 0
    for tensor, part in zip(tensors, raw):
        array = np.frombuffer(
            blob,
            dtype=_numpy_dtype(tensor.dtype),
            count=tensor.numel(),
            offset=offset,
        ).reshape(tuple(tensor.shape))
        arrays.append(array)
        offset += part.numel()

    return arrays


def _dense(tensor: torch.Tensor) -> torch.Tensor:
    # A 1-D view with stride 1, which a byte view needs. Not .contiguous():
    # PyTorch calls a length-1 slice contiguous whatever its stride (a real
    # model's class_id column of one detection has stride 6).
    flat = tensor.detach().reshape(-1)
    if flat.stride(0) == 1:
        return flat

    dense = torch.empty(flat.shape, dtype=flat.dtype, device=flat.device)
    dense.copy_(flat)

    return dense


def _numpy_dtype(dtype: torch.dtype) -> np.dtype:
    numpy_dtype = torch.empty((), dtype=dtype).numpy().dtype

    return numpy_dtype


def _class_and_confidence_labels(
    class_id: np.ndarray,
    confidence: np.ndarray,
    *,
    image_metadata: Optional[Mapping[str, Any]],
) -> Tuple[str, ...]:
    # The 08 _class_and_confidence_labels, on the host arrays: same texts.
    if len(class_id) == 0:
        return ()

    class_names = (image_metadata or {}).get("class_names")
    if class_names is None:
        raise ValueError("predictions.image_metadata has no class_names")

    labels = []
    for one_class, one_confidence in zip(class_id, confidence):
        if int(one_class) not in class_names:
            raise ValueError(f"class id {int(one_class)} has no name in class_names")
        labels.append(f"{class_names[int(one_class)]} {one_confidence:.2f}")
    labels = tuple(labels)

    return labels


class BatchedDetectionDrawingPrep(Block):
    """Prepare every image's detections of a run for drawing: one host transfer.

    The engine calls it once per run. ``prep_ms`` is the wall time of the
    whole batch, repeated in every row.
    """

    type = "structural/batched_detection_drawing_prep@v1"
    outputs = {
        "drawing": Output(
            DETECTION_DRAWING_KIND,
            source="predictions",
            description="New read-only host copy of boxes, classes and labels.",
        ),
        "prep_ms": Output(
            FLOAT_KIND,
            source="predictions",
            description="Transfer and label formatting wall time of the batch, ms.",
        ),
    }

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        predictions: Ref(
            OBJECT_DETECTION_PREDICTION_KIND, batch="always"  # noqa: F821
        ) = Field(description="Detections of every image, delivered as one batch.")

    def __init__(self):
        self._cuda = _ReadyOnReturn()

    def run(self, *, predictions: Batch) -> List[Dict[str, Any]]:
        """Copy every image's detections to the host in one transfer.

        Args:
            predictions: Detections of the run's images; not modified.

        Returns:
            One ``drawing`` / ``prep_ms`` result per image, in batch order.

        Raises:
            ValueError: As ``prepare_drawings``.
        """
        members = list(predictions)
        if not members:
            return []

        started = perf_counter()
        with self._cuda.work(members[0].xyxy.device):
            drawings = prepare_drawings(members)
        prep_ms = _elapsed_ms(started)
        results = [{"drawing": drawing, "prep_ms": prep_ms} for drawing in drawings]

        return results


class DrawingBoxVisualization(GpuBoxVisualization):
    """The 09 box painter, drawing from a ``DetectionDrawing``.

    Same copy, palette, thickness, ``gpu_draw_boxes`` and readiness wait.
    """

    type = "structural/box_visualization@v1"

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="RGB image to annotate.")
        drawing: Ref(DETECTION_DRAWING_KIND) = Field(
            description="Prepared boxes in the image's pixel coordinates."
        )

    def run(self, *, image: ImageData, drawing: DetectionDrawing) -> Dict[str, Any]:
        """Paint the box borders of ``drawing`` onto a copy of ``image``.

        Args:
            image: RGB image; not modified. Its pixels must be complete.
            drawing: Prepared detections of this image.

        Returns:
            ``image``: the painted copy, ready to read, same identity and
            provenance as ``image``; ``draw_ms``: wall time including the
            GPU work.

        Raises:
            ValueError: When the image is not RGB.
        """
        _require_rgb(image, block=self.type)

        started = perf_counter()
        with self._cuda.work(image.tensor_image.device):
            scene = image.tensor_image.clone(memory_format=torch.contiguous_format)
            if len(drawing) > 0:
                colors_rgb = np.asarray(
                    [PALETTE.by_idx(int(one)).as_rgb() for one in drawing.class_id],
                    dtype=np.uint8,
                )
                gpu_draw_boxes(
                    scene, drawing.xyxy.astype(int), colors_rgb, BOX_THICKNESS
                )
        result = {"image": image.with_pixels(scene), "draw_ms": _elapsed_ms(started)}

        return result


class DrawingLabelVisualization(GpuLabelVisualization):
    """The 09 label painter, drawing from a ``DetectionDrawing``, in place.

    Same placement, sprite cache, pinned paste ring and readiness wait. Bind
    an image this workflow owns, such as the box painter's output.
    """

    type = "structural/label_visualization@v1"

    class Params(BlockParams):
        image: Ref(IMAGE_KIND) = Field(description="RGB image changed in place.")
        drawing: Ref(DETECTION_DRAWING_KIND) = Field(
            description="Prepared label texts and boxes of this image."
        )

    def run(self, *, image: ImageData, drawing: DetectionDrawing) -> Dict[str, Any]:
        """Paste one label per detection above its box's top-left corner.

        Args:
            image: Contiguous RGB image; modified in place.
            drawing: Prepared detections of this image.

        Returns:
            ``image``: the same image object, ready to read; ``draw_ms``: wall
            time including the GPU work.

        Raises:
            ValueError: On a non-RGB or non-contiguous image.
        """
        _require_rgb(image, block=self.type)
        if not image.tensor_image.is_contiguous():
            raise ValueError(
                f"{self.type} paints in place and needs a contiguous image tensor; "
                "bind the box painter's output or another owned contiguous image"
            )

        started = perf_counter()
        scene = image.tensor_image
        with self._cuda.work(scene.device):
            sprites, origins = self._drawing_sprites(drawing, scene=scene)
            gpu_paste_label_sprites(scene, sprites, origins, self._table_ring)
        result = {"image": image, "draw_ms": _elapsed_ms(started)}

        return result

    def _drawing_sprites(
        self, drawing: DetectionDrawing, *, scene: torch.Tensor
    ) -> Tuple[List[_LabelSprite], List[Tuple[int, int]]]:
        # 09 _sprites_and_origins with the host arrays in place of .cpu() reads.
        frame_hw = (int(scene.shape[1]), int(scene.shape[2]))
        corners = drawing.xyxy[:, :2].astype(int)
        sprites: List[_LabelSprite] = []
        origins: List[Tuple[int, int]] = []
        for label, (x, y), class_id in zip(drawing.labels, corners, drawing.class_id):
            placed = self._place(
                label,
                anchor=(int(x), int(y)),
                background_bgr=PALETTE.by_idx(int(class_id)).as_bgr(),
                frame_hw=frame_hw,
                device=scene.device,
            )
            if placed is not None:
                sprites.append(placed[0])
                origins.append(placed[1])

        return sprites, origins


class BatchedDrawingBoxVisualization(Block):
    """``DrawingBoxVisualization`` for every image of a run, in one engine call."""

    type = "structural/batched_box_visualization@v1"
    outputs = DrawingBoxVisualization.outputs

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images to annotate, delivered as one batch."
        )
        drawing: Ref(DETECTION_DRAWING_KIND, batch="always") = Field(  # noqa: F821
            description="Prepared detections of every image, in batch order."
        )

    def __init__(self):
        self._painter = DrawingBoxVisualization()

    def run(self, *, image: Batch, drawing: Batch) -> List[Dict[str, Any]]:
        """Paint each image's boxes onto its copy, one image after another.

        Args:
            image: The run's RGB images; not modified.
            drawing: Their prepared detections, in the same order.

        Returns:
            One per-image result of ``DrawingBoxVisualization``, in batch order.

        Raises:
            ValueError: When batch lengths differ or an image is not RGB.
        """
        if len(image) != len(drawing):
            raise ValueError(
                f"Image and drawing batches must align: {len(image)} != {len(drawing)}"
            )

        results = [
            self._painter.run(image=member, drawing=member_drawing)
            for member, member_drawing in zip(image, drawing)
        ]

        return results


class BatchedDrawingLabelVisualization(Block):
    """``DrawingLabelVisualization`` for every image of a run, in one engine call."""

    type = "structural/batched_label_visualization@v1"
    mutates = ("image",)
    outputs = DrawingLabelVisualization.outputs

    class Params(BlockParams):
        # noqa F821: flake8 parses the string "always" in an annotation as a name.
        image: Ref(IMAGE_KIND, batch="always") = Field(  # noqa: F821
            description="RGB images changed in place, delivered as one batch."
        )
        drawing: Ref(DETECTION_DRAWING_KIND, batch="always") = Field(  # noqa: F821
            description="Prepared detections of every image, in batch order."
        )

    def __init__(self):
        self._painter = DrawingLabelVisualization()

    def run(self, *, image: Batch, drawing: Batch) -> List[Dict[str, Any]]:
        """Paste each image's labels in place, one image after another.

        Args:
            image: The run's contiguous RGB images; modified in place.
            drawing: Their prepared detections, in the same order.

        Returns:
            One per-image result of ``DrawingLabelVisualization``, in batch order.

        Raises:
            ValueError: When batch lengths differ, or an image is not RGB or contiguous.
        """
        if len(image) != len(drawing):
            raise ValueError(
                f"Image and drawing batches must align: {len(image)} != {len(drawing)}"
            )

        results = [
            self._painter.run(image=member, drawing=member_drawing)
            for member, member_drawing in zip(image, drawing)
        ]

        return results


BLOCKS = (
    *batched_blocks.BLOCKS,
    BatchedDetectionDrawingPrep,
    DrawingBoxVisualization,
    DrawingLabelVisualization,
    BatchedDrawingBoxVisualization,
    BatchedDrawingLabelVisualization,
)


def create_catalogue() -> Catalogue:
    """Collect ``BLOCKS``: the 10 blocks and the drawing blocks of this module.

    Compiles every workflow of ``drawing_backend.v2_workflow``, including the
    unchanged 10 ``per-painter`` one.

    Returns:
        A catalogue with the 10 detectors and painters, the drawing prep
        and the four drawing painters.
    """
    catalogue = Catalogue(
        BLOCKS,
        kinds=[
            IMAGE_KIND,
            OBJECT_DETECTION_PREDICTION_KIND,
            FLOAT_KIND,
            DETECTION_DRAWING_KIND,
        ],
    )

    return catalogue
