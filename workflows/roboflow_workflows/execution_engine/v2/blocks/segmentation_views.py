"""Low-resolution instance-segmentation view: a distinct payload and kind.

An instance-segmentation model predicts mask scores on a small grid (YOLOv8 and
RF-DETR: 1/4 of the network input per axis). The existing dense payload,
``InstanceDetections`` with a bool ``(N, H, W)`` mask, is produced by one
expensive step: bilinear upscale of every selected row to the image size, then
a threshold. A ``SegmentationView`` keeps the grid and postpones that step::

    model grid scores (N, mh, mw), padded like the network input
        │  low_res() / low_res_binary()   cheap; the grid in the image's place
        │  segmentation_view_ops          approximate area, zone, overlay
        └─ full_res()                     EXPENSIVE, cached once per view:
                                          inference_models'
                                          align_instance_segmentation_results,
                                          the call the model's dense
                                          post_process makes -> InstanceDetections

``full_res()`` returns the existing dense contract, so any consumer of
``instance_segmentation_prediction`` can use it. Direct grid use is a semantic
approximation (threshold before instead of after interpolation); the ops module
labels every such result.

Contracts:

=================  ==========================================================
rows               Row ``i`` of the scores belongs to row ``i`` of
                   ``detections`` (boxes in image pixels, class, confidence).
scores             ``logits`` (threshold usually 0.0), ``probabilities``
                   (usually 0.5) or ``binary`` (dense fallback). A cell or
                   pixel is foreground when ``score > threshold``.
geometry           ``MaskGridGeometry`` reproduces the reference helper,
                   including its rounded grid padding and static crop.
storage            The constructor copies the scores unless
                   ``take_ownership=True`` declares that nobody writes them
                   afterwards. Storage may then be borrowed and shared:
                   ``from_selected_rows`` passes its fresh gather,
                   ``with_threshold`` shares the parent view's scores, and
                   ``from_dense`` wraps the caller's dense mask. Sharing is
                   safe only because every holder treats it as immutable.
read-only          Attributes cannot be reassigned. Tensors are read-only by
                   contract; writes are not intercepted. To change masks,
                   build a new view (``select``, ``with_threshold``).
cache              ``full_res()`` computes at most once per view, also under
                   concurrent callers; a failure publishes nothing. The
                   threshold, geometry and rows are fixed per view, so the
                   cache needs no key. The cache lives as long as the view:
                   a retained view retains ``N x H x W`` bools.
CUDA streams       Build a view on the stream that produced its inputs, or on
                   one ordered after it. The view records a CUDA event there.
                   Every accessor (``detections``, ``scores``, ``low_res``,
                   ``full_res`` ...) makes the caller's current stream wait
                   for that event and marks the tensors as used by it, when
                   it is another stream. ``full_res()`` records a second
                   event after the upscale; later callers on other streams
                   wait for it. Results are ordered for the stream current
                   at the call. CPU views skip all of this.
dense fallback     ``from_dense`` wraps existing dense predictions when the
                   model offers no low-resolution stage. ``full_res()`` then
                   returns them unchanged and no upscale is saved. Its binary
                   threshold is fixed; ``select`` keeps the fallback.
=================  ==========================================================
"""

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple

import torch
from roboflow_workflows.execution_engine.v2.blocks.image_data import FrameMapping
from roboflow_workflows.execution_engine.v2.errors import ContractError
from roboflow_workflows.execution_engine.v2.kinds import Kind

from inference_models.entities import ImageDimensions
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
)

ScoreType = Literal["logits", "probabilities", "binary"]
SCORE_TYPES: Tuple[str, ...] = ("logits", "probabilities", "binary")

SizeHW = Tuple[int, int]


@dataclass(frozen=True)
class _StreamReadiness:
    """A CUDA event recorded on the stream that produced some tensors."""

    stream: Any
    event: Any

    @classmethod
    def record(cls, device: torch.device) -> Optional["_StreamReadiness"]:
        """Record an event on the current stream of ``device``; ``None`` on CPU."""
        if device.type != "cuda":
            return None

        stream = torch.cuda.current_stream(device)
        event = torch.cuda.Event()
        event.record(stream)

        return cls(stream=stream, event=event)


def _order_current_stream(
    readiness: Optional[_StreamReadiness], tensors: Sequence[torch.Tensor]
) -> None:
    """Order the caller's current stream after ``readiness``.

    No host synchronization. On another stream, the stream waits for the
    event, and the caching allocator learns that the stream uses the tensors,
    so their memory is not reused before that stream's work completes.
    """
    if readiness is None:
        return

    stream = torch.cuda.current_stream(readiness.stream.device)
    if stream == readiness.stream:
        return

    stream.wait_event(readiness.event)
    for tensor in tensors:
        tensor.record_stream(stream)


@dataclass(frozen=True)
class MaskGridGeometry:
    """Where a model's mask grid lies in the original image.

    The fields are the model's pre-processing record (``PreProcessingMetadata``)
    plus the grid size. Pre-processing runs this chain::

        original image (original_size_hw)
            ── static crop at (x, y) ──▶ crop (crop_size_hw, original pixels)
            ── × scale_xy, + padding ──▶ network input (inference_size_hw)
            ── ÷ stride ───────────────▶ grid (grid_size_hw)

    Without a static crop, the crop is the whole original image.

    Args:
        grid_size_hw: Padded grid size ``(mh, mw)``, as the model emits it.
        inference_size_hw: Network input size ``(H_in, W_in)``.
        padding_ltrb: Network-input padding ``(left, top, right, bottom)`` in
            network pixels; negative for a centre crop.
        scale_xy: Network pixels per crop pixel ``(x, y)``.
        crop_size_hw: Size of the static crop, before scaling and padding
            (``size_after_pre_processing``).
        original_size_hw: Size of the image the model received.
        static_crop_xywh: Static crop ``(x, y, width, height)`` in the original
            image; ``(0, 0, W, H)`` without a crop.

    Raises:
        ContractError: On non-positive sizes or scales, a crop that does not
            match ``crop_size_hw`` or leaves the image, and a partial crop at
            ``(0, 0)``: the reference helper places a crop on the full-image
            canvas only at a nonzero offset, so its dense mask would be
            crop-sized, not image-sized.
    """

    grid_size_hw: SizeHW
    inference_size_hw: SizeHW
    padding_ltrb: Tuple[int, int, int, int]
    scale_xy: Tuple[float, float]
    crop_size_hw: SizeHW
    original_size_hw: SizeHW
    static_crop_xywh: Tuple[int, int, int, int]

    def __post_init__(self) -> None:
        for name in (
            "grid_size_hw",
            "inference_size_hw",
            "crop_size_hw",
            "original_size_hw",
        ):
            size = getattr(self, name)
            if len(size) != 2 or min(size) <= 0:
                raise ContractError(f"{name} must be two positive sizes, got {size}")

        if len(self.scale_xy) != 2 or min(self.scale_xy) <= 0:
            raise ContractError(f"scale_xy must be positive, got {self.scale_xy}")

        crop_x, crop_y, crop_w, crop_h = self.static_crop_xywh
        image_h, image_w = self.original_size_hw
        if (crop_h, crop_w) != tuple(self.crop_size_hw):
            raise ContractError(
                f"static crop {self.static_crop_xywh} does not match "
                f"crop_size_hw {tuple(self.crop_size_hw)}"
            )
        if min(crop_x, crop_y) < 0 or crop_x + crop_w > image_w or crop_y + crop_h > image_h:
            raise ContractError(
                f"static crop {self.static_crop_xywh} leaves the "
                f"{image_w} x {image_h} image"
            )
        if (crop_x, crop_y) == (0, 0) and (crop_h, crop_w) != (image_h, image_w):
            raise ContractError(
                f"static crop {self.static_crop_xywh} at (0, 0) is smaller than "
                f"the {image_w} x {image_h} image; the reference helper returns "
                "crop-sized masks there, not full-image masks. Use a dense view."
            )

    @classmethod
    def from_pre_processing(
        cls, metadata: Any, *, grid_size_hw: SizeHW
    ) -> "MaskGridGeometry":
        """Build the geometry from one image's ``PreProcessingMetadata``.

        Args:
            metadata: ``inference_models`` ``PreProcessingMetadata`` of the image.
            grid_size_hw: Padded grid size the model emitted.

        Returns:
            The geometry.

        Raises:
            ContractError: When the metadata uses a non-square intermediate
                size; the reference helper is called with a different
                ``inference_size`` then, which this geometry does not model.
        """
        if getattr(metadata, "nonsquare_intermediate_size", None) is not None:
            raise ContractError(
                "nonsquare_intermediate_size pre-processing is not supported by "
                "MaskGridGeometry; use a dense view"
            )

        crop = metadata.static_crop_offset
        geometry = cls(
            grid_size_hw=tuple(grid_size_hw),
            inference_size_hw=(
                metadata.inference_size.height,
                metadata.inference_size.width,
            ),
            padding_ltrb=(
                metadata.pad_left,
                metadata.pad_top,
                metadata.pad_right,
                metadata.pad_bottom,
            ),
            scale_xy=(metadata.scale_width, metadata.scale_height),
            crop_size_hw=(
                metadata.size_after_pre_processing.height,
                metadata.size_after_pre_processing.width,
            ),
            original_size_hw=(
                metadata.original_size.height,
                metadata.original_size.width,
            ),
            static_crop_xywh=(
                crop.offset_x,
                crop.offset_y,
                crop.crop_width,
                crop.crop_height,
            ),
        )

        return geometry

    @classmethod
    def identity(cls, size_hw: SizeHW) -> "MaskGridGeometry":
        """Geometry of a grid that already is the image (dense fallback).

        Args:
            size_hw: Image size ``(H, W)``.

        Returns:
            A geometry without padding, scaling or crop.
        """
        height, width = size_hw
        geometry = cls(
            grid_size_hw=(height, width),
            inference_size_hw=(height, width),
            padding_ltrb=(0, 0, 0, 0),
            scale_xy=(1.0, 1.0),
            crop_size_hw=(height, width),
            original_size_hw=(height, width),
            static_crop_xywh=(0, 0, width, height),
        )

        return geometry

    @property
    def grid_padding_tblr(self) -> Tuple[int, int, int, int]:
        """Grid padding ``(top, bottom, left, right)``, rounded like the helper.

        Rounding makes mask and boxes disagree by up to half a cell when a
        network pad is not a multiple of the grid stride. The dense reference
        has the same offset; this geometry keeps it, so low-res and dense
        results differ only by representation.
        """
        grid_h, grid_w = self.grid_size_hw
        input_h, input_w = self.inference_size_hw
        left, top, right, bottom = self.padding_ltrb
        padding = (
            round(grid_h / input_h * top),
            round(grid_h / input_h * bottom),
            round(grid_w / input_w * left),
            round(grid_w / input_w * right),
        )

        return padding

    @property
    def unpadded_size_hw(self) -> SizeHW:
        """Size ``(h, w)`` of the grid part that covers the crop."""
        top, bottom, left, right = self.grid_padding_tblr
        grid_h, grid_w = self.grid_size_hw
        size = (grid_h - top - bottom, grid_w - left - right)

        return size

    def grid_to_image(self, frame_id: Optional[str] = None) -> FrameMapping:
        """Map unpadded grid cells to original-image pixels.

        Cell ``(u, v)`` covers ``x`` in ``[ox + u*sx, ox + (u+1)*sx)`` with
        ``sx = W_crop / w`` crop pixels per cell and ``ox`` the crop's x
        offset; the same for ``y``. Scales may differ per axis.

        Args:
            frame_id: Identity of the original image, when known.

        Returns:
            ``frame_xy = cell_xy * scale_xy + offset_xy``.
        """
        height, width = self.unpadded_size_hw
        crop_h, crop_w = self.crop_size_hw
        mapping = FrameMapping(
            frame_id=frame_id,
            frame_size_hw=self.original_size_hw,
            scale_xy=(crop_w / width, crop_h / height),
            offset_xy=(
                float(self.static_crop_xywh[0]),
                float(self.static_crop_xywh[1]),
            ),
        )

        return mapping

    def unpad(self, grid: torch.Tensor) -> torch.Tensor:
        """Cut the padding off a ``(N, mh, mw)`` grid, exactly as the helper does.

        Positive padding returns a slice (no copy). Negative padding (centre
        crop) zero-extends first, which allocates.

        Args:
            grid: Grid of this geometry's padded size.

        Returns:
            The ``(N, h, w)`` grid over the crop.
        """
        top, bottom, left, right = self.grid_padding_tblr
        if min(top, bottom, left, right) >= 0:
            unpadded = grid[
                :, top : grid.shape[1] - bottom, left : grid.shape[2] - right
            ]
            return unpadded

        extended = torch.nn.functional.pad(
            grid,
            (
                abs(min(left, 0)),
                abs(min(right, 0)),
                abs(min(top, 0)),
                abs(min(bottom, 0)),
            ),
            "constant",
            0,
        )
        top, bottom, left, right = (
            max(top, 0),
            max(bottom, 0),
            max(left, 0),
            max(right, 0),
        )
        unpadded = extended[
            :,
            top : extended.shape[1] - bottom,
            left : extended.shape[2] - right,
        ]

        return unpadded

    def reference_arguments(self) -> Dict[str, Any]:
        """Keyword arguments of ``align_instance_segmentation_results``."""
        crop_x, crop_y, crop_w, crop_h = self.static_crop_xywh
        arguments = {
            "padding": self.padding_ltrb,
            "scale_width": self.scale_xy[0],
            "scale_height": self.scale_xy[1],
            "original_size": ImageDimensions(*self.original_size_hw),
            "size_after_pre_processing": ImageDimensions(*self.crop_size_hw),
            "inference_size": ImageDimensions(*self.inference_size_hw),
            "static_crop_offset": StaticCropOffset(crop_x, crop_y, crop_w, crop_h),
        }

        return arguments


class SegmentationView:
    """Row-aligned low-resolution instance masks with an explicit dense boundary.

    See the module docstring for the contracts. Build model views with
    ``from_selected_rows`` and dense fallbacks with ``from_dense``.

    Args:
        detections: Selected instances: boxes in original-image pixels, class
            ids, confidences and metadata. The view takes them over.
        scores: ``(N, mh, mw)`` padded grid; floating for ``logits`` and
            ``probabilities``, bool or uint8 for ``binary``.
        score_type: Meaning of the scores.
        threshold: A cell or pixel is foreground when ``score > threshold``.
        geometry: Grid placement in the original image.
        take_ownership: ``True`` declares that nobody writes ``scores``
            afterwards; the view then borrows the storage instead of copying.

    Raises:
        ContractError: On mismatched rows, shapes, devices, dtypes or
            score semantics, or a ``binary`` threshold other than ``0.0``.
    """

    __slots__ = (
        "_detections",
        "_scores",
        "_score_type",
        "_threshold",
        "_geometry",
        "_dense_source",
        "_ready",
        "_full_res",
        "_lock",
        "_materializations",
    )

    def __init__(
        self,
        *,
        detections: Detections,
        scores: torch.Tensor,
        score_type: ScoreType,
        threshold: float,
        geometry: MaskGridGeometry,
        take_ownership: bool = False,
    ):
        _check_view_parts(
            detections=detections,
            scores=scores,
            score_type=score_type,
            threshold=threshold,
            geometry=geometry,
        )
        if not take_ownership:
            scores = scores.clone(memory_format=torch.contiguous_format)

        fields = {
            "_detections": detections,
            "_scores": scores,
            "_score_type": score_type,
            "_threshold": float(threshold),
            "_geometry": geometry,
            "_dense_source": None,
            "_ready": _StreamReadiness.record(scores.device),
            "_full_res": None,
            "_lock": threading.Lock(),
            "_materializations": 0,
        }
        for name, value in fields.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError(
            f"SegmentationView is read-only; cannot set {name!r}. Build a new "
            "view with select() or with_threshold()."
        )

    @classmethod
    def from_selected_rows(
        cls,
        all_scores: torch.Tensor,
        *,
        rows: torch.Tensor,
        detections: Detections,
        score_type: ScoreType,
        threshold: float,
        geometry: MaskGridGeometry,
    ) -> "SegmentationView":
        """Gather the selected rows of a model's grid output into a new view.

        The gather copies only the ``N`` selected rows into fresh storage, so
        the view never aliases a model output buffer that a later frame
        overwrites.

        Args:
            all_scores: ``(Q, mh, mw)`` grid of every candidate, any strides.
            rows: ``(N,)`` integer indices into ``all_scores``, in detection
                order; repeats and any order are allowed.
            detections: The ``N`` selected instances.
            score_type: Meaning of the scores.
            threshold: Foreground when ``score > threshold``.
            geometry: Grid placement in the original image.

        Returns:
            The view.

        Raises:
            ContractError: On mismatched rows, shapes or semantics; see
                ``select`` for the row checks.
        """
        _check_rows(rows, limit=int(all_scores.shape[0]))
        selected = all_scores.index_select(0, rows.to(all_scores.device))
        view = cls(
            detections=detections,
            scores=selected,
            score_type=score_type,
            threshold=threshold,
            geometry=geometry,
            take_ownership=True,
        )

        return view

    @classmethod
    def from_dense(cls, predictions: InstanceDetections) -> "SegmentationView":
        """Wrap dense predictions when no low-resolution stage is available.

        The grid is the image itself. The view borrows ``predictions`` and
        its mask without copying; both stay read-only. ``full_res()``
        returns ``predictions`` unchanged and counts no materialization;
        nothing is saved. The binary threshold is fixed at ``0.0``.

        Args:
            predictions: ``InstanceDetections`` with a dense ``(N, H, W)`` mask.

        Returns:
            The view.

        Raises:
            ContractError: On an RLE or non-binary mask.
        """
        mask = predictions.mask
        if not isinstance(mask, torch.Tensor) or mask.ndim != 3:
            raise ContractError(
                "from_dense needs a dense (N, H, W) mask, got " f"{type(mask).__name__}"
            )

        detections = Detections(
            xyxy=predictions.xyxy,
            class_id=predictions.class_id,
            confidence=predictions.confidence,
            image_metadata=predictions.image_metadata,
            bboxes_metadata=predictions.bboxes_metadata,
        )
        view = cls(
            detections=detections,
            scores=mask,
            score_type="binary",
            threshold=0.0,
            geometry=MaskGridGeometry.identity(tuple(mask.shape[1:])),
            take_ownership=True,
        )
        object.__setattr__(view, "_dense_source", predictions)

        return view

    def __len__(self) -> int:
        return int(self._scores.shape[0])

    def __repr__(self) -> str:
        return (
            f"SegmentationView(rows={len(self)}, grid={tuple(self._scores.shape[1:])}, "
            f"score_type={self._score_type!r}, threshold={self._threshold}, "
            f"dense_fallback={self.is_dense_fallback})"
        )

    @property
    def detections(self) -> Detections:
        """Selected instances, row-aligned with the scores. Read-only.

        Ordered for the caller's current CUDA stream.
        """
        self._order_inputs()

        return self._detections

    @property
    def scores(self) -> torch.Tensor:
        """Padded ``(N, mh, mw)`` grid as the model emitted it. Read-only.

        Ordered for the caller's current CUDA stream.
        """
        self._order_inputs()

        return self._scores

    @property
    def score_type(self) -> ScoreType:
        """``logits``, ``probabilities`` or ``binary``."""
        return self._score_type

    @property
    def threshold(self) -> float:
        """Foreground when ``score > threshold``."""
        return self._threshold

    @property
    def geometry(self) -> MaskGridGeometry:
        """Grid placement in the original image."""
        return self._geometry

    @property
    def is_dense_fallback(self) -> bool:
        """``True`` when the view wraps dense masks; no upscale can be saved."""
        return self._dense_source is not None

    @property
    def materialization_count(self) -> int:
        """Number of full-resolution upscales this view has computed."""
        return self._materializations

    def low_res(self) -> torch.Tensor:
        """Unpadded ``(N, h, w)`` scores over the crop. Read-only.

        Returns:
            A slice of the scores; a new tensor only for centre-crop padding.
        """
        unpadded = self._geometry.unpad(self.scores)

        return unpadded

    def low_res_binary(self) -> torch.Tensor:
        """Threshold the unpadded grid: ``(N, h, w)`` bool.

        Approximation: the dense reference interpolates first, then
        thresholds. Cells and full-resolution pixels disagree near
        boundaries.

        Returns:
            A new bool tensor.
        """
        if self._score_type == "binary":
            binary = self.low_res().bool()
            return binary

        binary = self.low_res() > self._threshold

        return binary

    def is_full_res_materialized(self) -> bool:
        """``True`` when ``full_res()`` has a cached result."""
        materialized = self._full_res is not None or self.is_dense_fallback

        return materialized

    def full_res(self) -> InstanceDetections:
        """Dense full-resolution predictions. EXPENSIVE on first call.

        Computes once per view and caches; concurrent callers wait for the
        one computation. A failure caches nothing, and a later call retries.
        The result is shared by every caller and read-only by contract; copy
        it before writing. On CUDA, the upscale runs on the first caller's
        current stream; every caller gets the result ordered for its own
        current stream, without a host synchronization.

        Returns:
            ``InstanceDetections`` with a bool ``(N, H, W)`` mask in original
            image pixels, equal to the model's dense post-processing of the
            same raw output. A dense fallback returns its wrapped predictions.
        """
        if self._dense_source is not None:
            self._order_inputs()
            return self._dense_source

        cached = self._full_res
        if cached is None:
            with self._lock:
                if self._full_res is None:
                    predictions = self._materialize()
                    ready = _StreamReadiness.record(predictions.mask.device)
                    object.__setattr__(self, "_full_res", (predictions, ready))
                cached = self._full_res

        predictions, ready = cached
        _order_current_stream(ready, _prediction_tensors(predictions))

        return predictions

    def compute_full_res(self) -> InstanceDetections:
        """Compute dense predictions without the cache. EXPENSIVE every call.

        For experiments that measure per-consumer cost; workflows use
        ``full_res()``. The upscale runs on the caller's current stream and
        its result belongs to that stream. A dense fallback returns its
        wrapped predictions and counts nothing.

        Returns:
            New ``InstanceDetections``; see ``full_res()``.
        """
        if self._dense_source is not None:
            self._order_inputs()
            return self._dense_source

        with self._lock:
            predictions = self._materialize()

        return predictions

    def _materialize(self) -> InstanceDetections:
        """One reference upscale on the current stream; the caller holds ``_lock``."""
        self._order_inputs()
        # The helper also rescales the boxes it gets, in place. The view's
        # boxes are already final, so it gets a throwaway float copy and its
        # box output is discarded.
        throwaway_boxes = self._detections.xyxy.to(dtype=torch.float32, copy=True)
        _, mask = align_instance_segmentation_results(
            image_bboxes=throwaway_boxes,
            masks=self._scores,
            binarization_threshold=self._threshold,
            **self._geometry.reference_arguments(),
        )
        object.__setattr__(self, "_materializations", self._materializations + 1)

        detections = self._detections
        predictions = InstanceDetections(
            xyxy=detections.xyxy,
            class_id=detections.class_id,
            confidence=detections.confidence,
            mask=mask,
            image_metadata=detections.image_metadata,
            bboxes_metadata=detections.bboxes_metadata,
        )

        return predictions

    def _order_inputs(self) -> None:
        """Order the current CUDA stream after the view's construction."""
        _order_current_stream(
            self._ready, [self._scores, *_detection_tensors(self._detections)]
        )

    def select(self, rows: torch.Tensor) -> "SegmentationView":
        """New view of the given rows, in the given order; repeats allowed.

        A dense fallback stays a dense fallback: its selected masks are
        copied, nothing is upscaled, and its threshold stays fixed.

        Args:
            rows: ``(K,)`` integer indices into this view.

        Returns:
            A view with its own score storage and its own empty cache.

        Raises:
            ContractError: On non-integer rows, and on out-of-range rows when
                ``rows`` is on the CPU. CUDA rows are not range-checked,
                because that needs a host synchronization; torch fails on
                an out-of-range CUDA row instead.
        """
        _check_rows(rows, limit=len(self))
        scores = self.scores
        device_rows = rows.to(scores.device)
        detections = self._detections
        metadata = detections.bboxes_metadata
        selected_detections = Detections(
            xyxy=detections.xyxy.index_select(0, device_rows),
            class_id=detections.class_id.index_select(0, device_rows),
            confidence=detections.confidence.index_select(0, device_rows),
            image_metadata=detections.image_metadata,
            bboxes_metadata=(
                None
                if metadata is None
                else [metadata[int(row)] for row in rows.tolist()]
            ),
        )
        if self._dense_source is not None:
            selected_predictions = InstanceDetections(
                xyxy=selected_detections.xyxy,
                class_id=selected_detections.class_id,
                confidence=selected_detections.confidence,
                mask=scores.index_select(0, device_rows),
                image_metadata=selected_detections.image_metadata,
                bboxes_metadata=selected_detections.bboxes_metadata,
            )
            return SegmentationView.from_dense(selected_predictions)

        view = SegmentationView(
            detections=selected_detections,
            scores=scores.index_select(0, device_rows),
            score_type=self._score_type,
            threshold=self._threshold,
            geometry=self._geometry,
            take_ownership=True,
        )

        return view

    def with_threshold(self, threshold: float) -> "SegmentationView":
        """New view with another threshold; borrows the read-only scores.

        Args:
            threshold: Foreground when ``score > threshold``.

        Returns:
            A view with its own empty cache.

        Raises:
            ContractError: For ``binary`` scores, including dense fallbacks
                (their threshold is fixed at ``0.0``), or an invalid
                probability threshold.
        """
        if self._score_type == "binary":
            raise ContractError(
                "binary scores have a fixed threshold; a dense fallback or "
                "binary view cannot be re-thresholded"
            )

        view = SegmentationView(
            detections=self.detections,
            scores=self._scores,
            score_type=self._score_type,
            threshold=threshold,
            geometry=self._geometry,
            take_ownership=True,
        )

        return view

    def describe(self) -> Dict[str, Any]:
        """JSON-friendly summary for logs and experiment records.

        Returns:
            Rows, grid sizes, score semantics, storage bytes and cache state.
        """
        scores = self._scores
        summary = {
            "rows": len(self),
            "grid_size_hw": list(self._geometry.grid_size_hw),
            "unpadded_size_hw": list(self._geometry.unpadded_size_hw),
            "original_size_hw": list(self._geometry.original_size_hw),
            "score_type": self._score_type,
            "threshold": self._threshold,
            "score_dtype": str(scores.dtype),
            "score_bytes": scores.numel() * scores.element_size(),
            "device": str(scores.device),
            "dense_fallback": self.is_dense_fallback,
            "materializations": self._materializations,
        }

        return summary


def _check_view_parts(
    *,
    detections: Any,
    scores: Any,
    score_type: Any,
    threshold: Any,
    geometry: Any,
) -> None:
    """Metadata-only checks; reads no tensor values."""
    if not isinstance(detections, Detections):
        raise ContractError(
            f"detections must be inference_models.Detections, got "
            f"{type(detections).__name__}"
        )
    if not isinstance(geometry, MaskGridGeometry):
        raise ContractError(
            f"geometry must be MaskGridGeometry, got {type(geometry).__name__}"
        )
    if score_type not in SCORE_TYPES:
        raise ContractError(
            f"score_type must be one of {SCORE_TYPES}, got {score_type!r}"
        )
    if not isinstance(scores, torch.Tensor) or scores.ndim != 3:
        raise ContractError(
            "scores must be an (N, mh, mw) torch.Tensor, got "
            f"{getattr(scores, 'shape', type(scores).__name__)}"
        )

    is_binary = scores.dtype in (torch.bool, torch.uint8)
    if score_type == "binary" and not is_binary:
        raise ContractError(f"binary scores must be bool or uint8, got {scores.dtype}")
    if score_type == "binary" and threshold != 0.0:
        raise ContractError(f"binary scores use the fixed threshold 0.0, got {threshold}")
    if score_type != "binary" and not scores.dtype.is_floating_point:
        raise ContractError(f"{score_type} scores must be floating, got {scores.dtype}")
    if score_type == "probabilities" and not 0.0 <= threshold <= 1.0:
        raise ContractError(f"probability threshold must be in [0, 1], got {threshold}")

    xyxy = detections.xyxy
    if not isinstance(xyxy, torch.Tensor) or xyxy.ndim != 2 or xyxy.shape[1] != 4:
        raise ContractError(
            "detections.xyxy must be an (N, 4) torch.Tensor, got "
            f"{getattr(xyxy, 'shape', type(xyxy).__name__)}"
        )

    rows = int(xyxy.shape[0])
    for name in ("class_id", "confidence"):
        value = getattr(detections, name)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != (rows,):
            raise ContractError(
                f"detections.{name} must be a ({rows},) torch.Tensor, got "
                f"{getattr(value, 'shape', type(value).__name__)}"
            )

    metadata = detections.bboxes_metadata
    if metadata is not None and len(metadata) != rows:
        raise ContractError(
            f"detections.bboxes_metadata holds {len(metadata)} rows for {rows} boxes"
        )
    if scores.shape[0] != rows:
        raise ContractError(f"scores hold {scores.shape[0]} rows for {rows} detections")
    if tuple(scores.shape[1:]) != tuple(geometry.grid_size_hw):
        raise ContractError(
            f"scores grid {tuple(scores.shape[1:])} does not match geometry "
            f"{tuple(geometry.grid_size_hw)}"
        )
    for tensor in _detection_tensors(detections):
        if tensor.device != scores.device:
            raise ContractError(
                f"scores are on {scores.device}, detections on {tensor.device}"
            )


def _check_rows(rows: Any, *, limit: int) -> None:
    """Rows are a 1-D integer tensor; CPU rows also lie in ``[0, limit)``."""
    if (
        not isinstance(rows, torch.Tensor)
        or rows.ndim != 1
        or rows.dtype.is_floating_point
        or rows.dtype.is_complex
        or rows.dtype == torch.bool
    ):
        raise ContractError(
            "rows must be a 1-D integer tensor, got "
            f"{getattr(rows, 'dtype', type(rows).__name__)} "
            f"{tuple(getattr(rows, 'shape', ()))}"
        )
    if rows.device.type != "cpu" or rows.numel() == 0:
        return

    low, high = int(rows.min()), int(rows.max())
    if low < 0 or high >= limit:
        raise ContractError(f"rows must lie in [0, {limit}), got [{low}, {high}]")


def _detection_tensors(detections: Detections) -> List[torch.Tensor]:
    return [detections.xyxy, detections.class_id, detections.confidence]


def _prediction_tensors(predictions: InstanceDetections) -> List[torch.Tensor]:
    return [
        predictions.mask,
        predictions.xyxy,
        predictions.class_id,
        predictions.confidence,
    ]


def _validate_view(payload: Any) -> bool:
    if not isinstance(payload, SegmentationView):
        raise ContractError(
            f"Expected SegmentationView, got {type(payload).__name__}. Dense "
            "InstanceDetections belong to instance_segmentation_prediction."
        )

    return True


def _deserialize_view(value: Any) -> SegmentationView:
    _validate_view(value)

    return value


def _refuse_serialization(payload: Any) -> Any:
    raise ContractError(
        "instance_segmentation_view has no wire format. Serialize full_res() "
        "through instance_segmentation_prediction; that is the explicit, "
        "expensive dense boundary."
    )


INSTANCE_SEGMENTATION_VIEW_KIND = Kind(
    name="instance_segmentation_view",
    description=(
        "SegmentationView: low-resolution mask scores row-aligned with "
        "detections; full_res() gives instance_segmentation_prediction."
    ),
    validate=_validate_view,
    deserialize=_deserialize_view,
    serialize=_refuse_serialization,
)

__all__ = [
    "INSTANCE_SEGMENTATION_VIEW_KIND",
    "MaskGridGeometry",
    "SCORE_TYPES",
    "ScoreType",
    "SegmentationView",
]
