"""Lazy, row-aligned instance masks in image coordinates.

``from_predictions`` accepts the native score-grid carrier and dense/RLE
fallbacks. Score grids retain unthresholded values: accurate materialization
unpads, resizes, thresholds, then pastes. Reduced binary rasters retain their
encoded size and cannot reconstruct that accurate result; providers offer an
explicit coarse expansion. Selection and metadata updates never materialize.

Views borrow producer-owned immutable snapshots. Direct construction copies
scores unless ownership is explicitly transferred. Full-image reconstruction
is cached once per view; failures retry. CUDA readiness and cache publication
order each caller's current stream and retain tensor storage without a host
synchronization. Other resolution factors compute on the caller's stream.
"""

import threading
from dataclasses import dataclass, replace
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.image import FrameMapping
from roboflow_workflows.v2.data.kinds import Kind
from roboflow_workflows.v2.data.masks.stream_order import (
    StreamOrderedOnce,
    StreamReadiness,
    order_current_stream,
)
from roboflow_workflows.v2.data.prediction_ops import RowSelection, take_rows
from roboflow_workflows.v2.data.row_identity import checked_row_record

from inference_models.entities import ImageDimensions
from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import (
    InstanceSegmentationGrid,
    InstancesRLEMasks,
)
from inference_models.models.common.roboflow.model_packages import StaticCropOffset
from inference_models.models.common.roboflow.post_processing import (
    align_instance_segmentation_results,
    resolve_mask_output_size,
    resolve_mask_target_size,
)

ScoreType = Literal["logits", "probabilities", "binary"]
SCORE_TYPES: Tuple[str, ...] = ("logits", "probabilities", "binary")

SizeHW = Tuple[int, int]


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
            match ``crop_size_hw`` or leaves the image.
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
        if (
            min(crop_x, crop_y) < 0
            or crop_x + crop_w > image_w
            or crop_y + crop_h > image_h
        ):
            raise ContractError(
                f"static crop {self.static_crop_xywh} leaves the "
                f"{image_w} x {image_h} image"
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

        Non-square intermediate dimensions take precedence over network size,
        matching the native post-processing helper.
        """
        inference_size = metadata.nonsquare_intermediate_size or metadata.inference_size

        crop = metadata.static_crop_offset
        geometry = cls(
            grid_size_hw=tuple(grid_size_hw),
            inference_size_hw=(
                inference_size.height,
                inference_size.width,
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
        "_decoded",
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

        self._initialize(detections, scores, score_type, threshold, geometry)

    def _initialize(self, detections, scores, score_type, threshold, geometry):
        lock = threading.Lock()
        fields = {
            "_detections": detections,
            "_scores": scores,
            "_score_type": score_type,
            "_threshold": float(threshold),
            "_geometry": geometry,
            "_dense_source": None,
            "_ready": StreamReadiness.record(detections.xyxy.device),
            "_full_res": StreamOrderedOnce(lock),
            "_decoded": StreamOrderedOnce(),
            "_lock": lock,
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
        selection = _row_selection(rows, count=int(all_scores.shape[0]))
        # Gather even an identity selection: RowSelection.take would alias it.
        positions = selection.positions_on(all_scores.device)
        selected = all_scores.index_select(0, positions)
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
    def from_predictions(cls, predictions: InstanceDetections) -> "SegmentationView":
        """Adapt a native grid, dense raster or RLE without materializing masks.

        Grid boxes already use image coordinates. Binary carriers store boxes
        in their encoded raster coordinates; the view scales those boxes to the
        image frame while retaining the original mask and its size metadata.
        Reduced binary data cannot reconstruct an accurate full-image mask.
        """
        from roboflow_workflows.v2.data.predictions import (
            INSTANCE_SEGMENTATION_PREDICTION_KIND,
        )

        INSTANCE_SEGMENTATION_PREDICTION_KIND.check(predictions)
        mask = predictions.mask
        if isinstance(mask, InstanceSegmentationGrid):
            image_h, image_w = predictions.image_size
            x, y, width, height = mask.image_crop_xywh
            geometry = MaskGridGeometry(
                grid_size_hw=tuple(mask.scores.shape[1:]),
                inference_size_hw=tuple(mask.scores.shape[1:]),
                padding_ltrb=mask.grid_padding_ltrb,
                scale_xy=(1.0, 1.0),
                crop_size_hw=(height, width),
                original_size_hw=(image_h, image_w),
                static_crop_xywh=(x, y, width, height),
            )
            return cls(
                detections=image_detections_of(predictions),
                scores=mask.scores,
                score_type=mask.score_type,
                threshold=mask.threshold,
                geometry=geometry,
                take_ownership=True,
            )

        encoded, frame, image = binary_mask_sizes(predictions)
        geometry = MaskGridGeometry(
            grid_size_hw=encoded,
            inference_size_hw=encoded,
            padding_ltrb=(0, 0, 0, 0),
            scale_xy=(1.0, 1.0),
            crop_size_hw=frame,
            original_size_hw=image,
            static_crop_xywh=(0, 0, frame[1], frame[0]),
        )
        view = cls.__new__(cls)
        view._initialize(
            image_detections_of(predictions),
            mask if isinstance(mask, torch.Tensor) else None,
            "binary",
            0.0,
            geometry,
        )
        object.__setattr__(view, "_dense_source", predictions)
        return view

    @classmethod
    def from_dense(cls, predictions: InstanceDetections) -> "SegmentationView":
        """Wrap a dense binary carrier, preserving its encoded and image sizes."""
        if not isinstance(predictions.mask, torch.Tensor):
            raise ContractError("from_dense needs a dense (N, H, W) mask")
        return cls.from_predictions(predictions)

    @property
    def binary_source(self) -> Optional[InstanceDetections]:
        """Original dense/RLE carrier, or None for a score grid. Read-only."""
        self._order_inputs()
        return self._dense_source

    def __len__(self) -> int:
        return int(self._detections.xyxy.shape[0])

    def __repr__(self) -> str:
        return (
            f"SegmentationView(rows={len(self)}, grid={self._geometry.grid_size_hw}, "
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

        if self._scores is None:
            return self._decoded.get(self._decode_source).mask

        return self._scores

    @property
    def score_type(self) -> ScoreType:
        """``logits``, ``probabilities`` or ``binary``."""
        return self._score_type

    @property
    def score_dtype(self) -> torch.dtype:
        """Dtype of the stored scores; reads no values and orders no stream."""
        return self._scores.dtype if self._scores is not None else torch.bool

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
        """``True`` when the view wraps binary dense/RLE data without scores."""
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
        materialized = self._full_res.is_set or (
            self._dense_source is not None
            and isinstance(self._dense_source.mask, torch.Tensor)
            and self._geometry.grid_size_hw == self._geometry.original_size_hw
        )

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
            same prepared scores. Boxes are clipped and rounded in image pixels.
            Reduced binary fallbacks raise because scores were discarded.
        """
        if self._dense_source is not None:
            self._order_inputs()
            return self._full_res.get(self._binary_full_res)

        predictions = self._full_res.get(self._materialize)

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
            return self._full_res.get(self._binary_full_res)

        with self._lock:
            predictions = self._materialize()

        return predictions

    def materialize(
        self, *, masks_resolution_factor: float = 1.0
    ) -> InstanceDetections:
        """Resize scores before thresholding at the requested native factor.

        Factor 1 uses image pixels and the shared full-resolution cache. Factor
        0 uses the unpadded score-grid size; intermediate factors follow the
        model library's interpolation and crop-placement rules. Lower factors
        need not be smaller when the source image is smaller than the grid.
        Binary fallbacks support factor 1 only, when it is reconstructable.
        """
        if not 0.0 <= masks_resolution_factor <= 1.0:
            raise ContractError("masks_resolution_factor must be finite and in [0, 1]")
        if masks_resolution_factor == 1.0:
            return self.full_res()
        if self._score_type == "binary":
            raise ContractError("resolution factors require unthresholded scores")
        with self._lock:
            return self._materialize(masks_resolution_factor)

    def to_rle(self, *, masks_resolution_factor: float = 1.0) -> InstanceDetections:
        """Materialize masks, then encode COCO RLE at an explicit host boundary.

        Encoding reads mask runs on the CPU and therefore synchronizes a CUDA
        source. Raster, image and frame sizes, rows and metadata are preserved.
        """
        from inference_models.models.common.rle_utils import torch_mask_to_coco_rle

        dense = self.materialize(masks_resolution_factor=masks_resolution_factor)
        masks = InstancesRLEMasks(
            image_size=dense.image_size or tuple(dense.mask.shape[1:]),
            mask_size=dense.mask_size,
            masks=[torch_mask_to_coco_rle(mask)["counts"] for mask in dense.mask],
        )
        return replace(dense, mask=masks)

    def _decode_source(self) -> InstanceDetections:
        from roboflow_workflows.v2.data.rle import decode_rle_tensor

        source = self._dense_source
        mask = decode_rle_tensor(source.mask, device=source.xyxy.device)
        return replace(source, mask=mask)

    def _binary_full_res(self) -> InstanceDetections:
        source = self._dense_source
        encoded, frame, _ = binary_mask_sizes(source)
        if encoded != frame:
            raise ContractError(
                "reduced binary masks cannot reconstruct accurate full-image masks; "
                "request coarse fidelity or retain an unthresholded score grid"
            )
        if isinstance(source.mask, InstancesRLEMasks):
            source = self._decoded.get(self._decode_source)
        return binary_predictions_in_image(source)

    def _materialize(self, masks_resolution_factor: float = 1.0) -> InstanceDetections:
        """One score reconstruction on the current stream; caller holds the lock."""
        self._order_inputs()
        geometry = self._geometry
        arguments = geometry.reference_arguments()
        rows = len(self)
        throwaway_boxes = torch.zeros((rows, 4), dtype=torch.float32, device="cpu")
        if rows:
            _, mask = align_instance_segmentation_results(
                image_bboxes=throwaway_boxes,
                masks=self._scores,
                binarization_threshold=self._threshold,
                masks_resolution_factor=masks_resolution_factor,
                **arguments,
            )
        else:
            size = resolve_mask_output_size(
                *geometry.grid_size_hw,
                **{
                    key: arguments[key]
                    for key in (
                        "padding",
                        "inference_size",
                        "original_size",
                        "size_after_pre_processing",
                        "static_crop_offset",
                    )
                },
                masks_resolution_factor=masks_resolution_factor,
            )
            mask = torch.empty((0, *size), dtype=torch.bool, device=self._scores.device)
        target_h, target_w = resolve_mask_target_size(
            *geometry.unpadded_size_hw,
            size_after_pre_processing=arguments["size_after_pre_processing"],
            masks_resolution_factor=masks_resolution_factor,
        )
        x, y, crop_w, crop_h = geometry.static_crop_xywh
        scale = self._detections.xyxy.new_tensor(
            [target_w / crop_w, target_h / crop_h] * 2, dtype=torch.float32
        )
        origin = scale.new_tensor([x, y] * 2)
        offset = scale.new_tensor(
            [round(x * target_w / crop_w), round(y * target_h / crop_h)] * 2
        )
        boxes = (self._detections.xyxy.float() - origin) * scale + offset
        boxes[:, 0::2].clamp_(0, mask.shape[2])
        boxes[:, 1::2].clamp_(0, mask.shape[1])
        if masks_resolution_factor == 1.0:
            boxes = boxes.round().int()
        object.__setattr__(self, "_materializations", self._materializations + 1)
        detections = self._detections
        return InstanceDetections(
            xyxy=boxes,
            class_id=detections.class_id,
            confidence=detections.confidence,
            mask=mask,
            image_metadata=detections.image_metadata,
            bboxes_metadata=detections.bboxes_metadata,
            mask_size=tuple(mask.shape[1:]),
            image_size=geometry.original_size_hw,
            mask_frame_size=geometry.original_size_hw,
        )

    def _order_inputs(self) -> None:
        """Order the current CUDA stream after the view's construction."""
        order_current_stream(
            self._ready,
            [
                tensor
                for tensor in (
                    self._scores,
                    self._dense_source.xyxy if self._dense_source is not None else None,
                    *_detection_tensors(self._detections),
                )
                if tensor is not None
            ],
        )

    def select(self, rows: Union[torch.Tensor, RowSelection]) -> "SegmentationView":
        """New view of the given rows, in the given order; repeats allowed.

        The native ``take_rows`` helper selects the detections, so boxes,
        classes, confidences, row metadata (copied dicts) and scores follow
        one selection. A dense fallback stays a dense fallback: its selected
        masks are gathered, nothing is upscaled, and its threshold stays fixed.

        Args:
            rows: ``(K,)`` integer indices into this view, or a prepared
                ``RowSelection`` of this view's rows.

        Returns:
            A view with its own empty cache. Its storage is new, except for an
            identity selection, which shares the read-only tensors.

        Raises:
            ContractError: On rows that are not a tensor or fail
                ``RowSelection.of``: not 1-D, not an integer dtype it accepts,
                or out of range on the CPU. CUDA rows are not range-checked,
                because that needs a host synchronization. An out-of-range
                or negative CUDA row triggers a device-side assert. It raises
                ``RuntimeError``, not ``ContractError``, in the call or at a
                later synchronizing operation, and every later CUDA
                operation in the process fails until the process restarts.
        """
        if isinstance(rows, RowSelection):
            selection = rows
        else:
            selection = _row_selection(rows, count=len(self))
        self._order_inputs()

        if self._dense_source is not None:
            selected_predictions = take_rows(self._dense_source, selection)
            return SegmentationView.from_predictions(selected_predictions)

        selected_detections = take_rows(self._detections, selection)
        view = SegmentationView(
            detections=selected_detections,
            scores=selection.take(self.scores),
            score_type=self._score_type,
            threshold=self._threshold,
            geometry=self._geometry,
            take_ownership=True,
        )

        return view

    def with_row_metadata(
        self, rows_metadata: Optional[List[dict]]
    ) -> "SegmentationView":
        """New view with replaced row dictionaries and the same read-only storage.

        Shallow-copy the supplied dictionaries. Scores, detection tensors,
        image metadata, geometry and threshold remain unchanged; no mask is
        copied or materialized. The new view starts with its own empty cache.
        Dense fallbacks remain dense fallbacks.

        Args:
            rows_metadata: One dictionary per row, or None to remove row metadata.

        Returns:
            A new view borrowing this view's tensors, with fresh row dictionaries.

        Raises:
            ContractError: When metadata is not a list of dictionaries with one
                entry per row, or None.
        """
        copied_rows = None
        if rows_metadata is not None:
            if not isinstance(rows_metadata, list):
                raise ContractError(
                    "rows_metadata must be a list of dictionaries or None"
                )
            if len(rows_metadata) != len(self):
                raise ContractError(
                    f"rows_metadata holds {len(rows_metadata)} rows "
                    f"for {len(self)} instances"
                )
            if any(not isinstance(row, dict) for row in rows_metadata):
                raise ContractError("rows_metadata must contain only dictionaries")

            copied_rows = [dict(row) for row in rows_metadata]

        self._order_inputs()
        if self._dense_source is not None:
            predictions = replace(self._dense_source, bboxes_metadata=copied_rows)
            view = SegmentationView.from_predictions(predictions)
        else:
            detections = replace(self._detections, bboxes_metadata=copied_rows)
            view = SegmentationView(
                detections=detections,
                scores=self._scores,
                score_type=self._score_type,
                threshold=self._threshold,
                geometry=self._geometry,
                take_ownership=True,
            )

        return view

    def with_image_metadata(self, image_metadata: Optional[dict]) -> "SegmentationView":
        """Replace image/row-identity metadata without copying or decoding masks."""
        self._order_inputs()
        if self._dense_source is not None:
            return SegmentationView.from_predictions(
                replace(self._dense_source, image_metadata=image_metadata)
            )
        return SegmentationView(
            detections=replace(self._detections, image_metadata=image_metadata),
            scores=self._scores,
            score_type=self._score_type,
            threshold=self._threshold,
            geometry=self._geometry,
            take_ownership=True,
        )

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
            "score_dtype": str(self.score_dtype),
            "score_bytes": (
                scores.numel() * scores.element_size() if scores is not None else None
            ),
            "device": str(self._detections.xyxy.device),
            "dense_fallback": self.is_dense_fallback,
            "materializations": self._materializations,
        }

        return summary


def binary_mask_sizes(predictions: InstanceDetections) -> Tuple[SizeHW, SizeHW, SizeHW]:
    """Encoded raster, represented image extent, and original image sizes."""
    mask = predictions.mask
    native = (
        tuple(mask.mask_size or mask.image_size)
        if isinstance(mask, InstancesRLEMasks)
        else tuple(mask.shape[1:])
    )
    encoded = tuple(predictions.mask_size or native)
    image = tuple(
        predictions.image_size
        or (mask.image_size if isinstance(mask, InstancesRLEMasks) else native)
    )
    frame = tuple(predictions.mask_frame_size or image)
    if encoded != native or any(
        len(size) != 2 or min(size) <= 0 for size in (encoded, frame, image)
    ):
        raise ContractError("mask_size must match the positive encoded raster size")
    if any(extent > total for extent, total in zip(frame, image)):
        raise ContractError("mask_frame_size must fit inside image_size")
    return encoded, frame, image


def image_detections_of(predictions: InstanceDetections) -> Detections:
    """Read image-coordinate boxes without materializing any masks."""
    boxes = predictions.xyxy
    if predictions.mask is not None and not isinstance(
        predictions.mask, InstanceSegmentationGrid
    ):
        encoded, frame, _ = binary_mask_sizes(predictions)
        if encoded != frame:
            boxes = boxes.float() * boxes.new_tensor(
                [frame[1] / encoded[1], frame[0] / encoded[0]] * 2,
                dtype=torch.float32,
            )
    return Detections(
        xyxy=boxes,
        class_id=predictions.class_id,
        confidence=predictions.confidence,
        image_metadata=predictions.image_metadata,
        bboxes_metadata=predictions.bboxes_metadata,
    )


def image_predictions_of(predictions: InstanceDetections) -> InstanceDetections:
    """Normalize masks and boxes to image pixels, preserving full-size RLE."""
    if predictions.mask is None:
        return predictions
    if isinstance(predictions.mask, InstanceSegmentationGrid):
        return SegmentationView.from_predictions(predictions).full_res()
    encoded, frame, image = binary_mask_sizes(predictions)
    if encoded == frame == image:
        return predictions
    return binary_predictions_in_image(predictions)


def binary_predictions_in_image(
    predictions: InstanceDetections, *, approximate: bool = False
) -> InstanceDetections:
    """Place binary pixels in image coordinates; reduced rasters need coarse consent.

    Resizing binary pixels cannot recover the scores discarded by thresholding.
    Exact placement is possible when the encoded raster equals its frame extent.
    """
    encoded, frame, image = binary_mask_sizes(predictions)
    if encoded != frame and not approximate:
        raise ContractError(
            "reduced binary masks cannot reconstruct accurate full-image masks; "
            "request coarse fidelity or retain an unthresholded score grid"
        )
    mask = predictions.mask
    if isinstance(mask, InstancesRLEMasks):
        from roboflow_workflows.v2.data.rle import decode_rle_tensor

        mask = decode_rle_tensor(mask, device=predictions.xyxy.device)
    if encoded == frame == image and mask is predictions.mask:
        return predictions
    if encoded != frame:
        if len(mask):
            mask = torch.nn.functional.interpolate(
                mask[:, None].float(),
                size=frame,
                mode="nearest-exact",
            )[:, 0].bool()
        else:
            mask = torch.empty((0, *frame), dtype=torch.bool, device=mask.device)
    if frame != image:
        canvas = torch.zeros((len(mask), *image), dtype=mask.dtype, device=mask.device)
        canvas[:, : frame[0], : frame[1]] = mask
        mask = canvas
    return replace(
        predictions,
        xyxy=image_detections_of(predictions).xyxy,
        mask=mask,
        mask_size=image,
        mask_frame_size=image,
        image_size=image,
    )


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
        raise ContractError(
            f"binary scores use the fixed threshold 0.0, got {threshold}"
        )
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


def _row_selection(rows: Any, *, count: int) -> RowSelection:
    """Tensor rows as a ``RowSelection``; ``RowSelection.of`` checks the values."""
    if not isinstance(rows, torch.Tensor):
        raise ContractError(f"rows must be a torch.Tensor, got {type(rows).__name__}")

    selection = RowSelection.of(count, indices=rows)

    return selection


def _detection_tensors(detections: Detections) -> List[torch.Tensor]:
    """Row tensors of the detections, including row-identity ids when present.

    Raises:
        ContractError: When the detections hold a malformed or stale row record.
    """
    tensors = [detections.xyxy, detections.class_id, detections.confidence]
    record = checked_row_record(
        detections.image_metadata,
        rows=int(detections.xyxy.shape[0]),
        device=detections.xyxy.device,
        name="detections.image_metadata",
    )
    if record is not None and record.ids is not None:
        tensors.append(record.ids)

    return tensors


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
