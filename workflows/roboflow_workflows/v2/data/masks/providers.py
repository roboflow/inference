"""Instance-mask providers: three representations behind one binary request.

A provider owns row-aligned detections and their masks, in the form the model
produced them. Consumers ask for binary masks with a fidelity; the provider
answers with the representation's own reference result, or refuses::

    provider                    wraps                          score grid
    GridMaskProvider            SegmentationView (model grid)  yes, float scores
    DenseMaskProvider           InstanceDetections, (N, H, W)  no
    RLEMaskProvider             InstanceDetections, COCO RLE   no

    provider.binary(fidelity="reference")    the reference result, never a preview
    provider.binary(fidelity="approximate")  grid: nearest_full_res, labelled
                                             approximate; dense/RLE: their
                                             reference, labelled reference

Every ``BinaryMasks`` names the ``MaskRecipe`` it equals or approximates: the
native producer, score domain and dtype, threshold and comparison, geometry
and the actual operation order. The recipe describes; it is not executed.

The representations are not interchangeable. Dense and RLE inputs carry no
scores, so ``score_grid()`` and ``with_threshold()`` refuse instead of
inventing logits. A grid is exact only for its stored scores: a model that
returns clamped or otherwise altered low-resolution scores (SAM2 clamps its
logits) is not reproduced by thresholding them, so such models belong in a
dense provider. Prototype bases with coefficients, semantic label grids and
prompt-model auxiliaries have no provider here.

Rows and frames:

=================  ==========================================================
selection          ``select`` builds one ``RowSelection`` and applies it to
                   every aligned field with the native ``take_rows`` helper:
                   order and repeats kept, empty selections keep ``(0, ...)``
                   shapes. RLE selection copies no bytes objects.
frame              ``image_metadata`` is shared by selections, so a selection
                   keeps the frame. ``frame`` reads it with ``frame_of`` and
                   checks that the mask raster has the frame's size. It never
                   derives geometry from a mask shape.
root coordinates   Results are in the frame the model saw. Mapping to the
                   workflow root stays the separate, explicit conversion.
=================  ==========================================================

Caches: each provider computes a result at most once, also under concurrent
callers; a failure publishes nothing and a later call retries. The grid view
and the providers share one cache helper, ``stream_order.StreamOrderedOnce``,
which also orders every caller's CUDA stream after the result. Selection and
threshold changes build a new provider with empty caches. Results are shared
and read-only by contract.

Host work: grid and dense providers read no tensor values on the host. RLE
decoding is CPU work by nature: the counts are validated in Python, decoded by
pycocotools into ``(H, W, N)`` uint8, transposed into one ``(N, H, W)`` bool
array and, for detections on an accelerator, copied there once; the pageable
copy makes the host wait for the current stream.
"""

from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Tuple, Union

import torch
from roboflow_workflows.v2.data.errors import ContractError
from roboflow_workflows.v2.data.frames import FrameRef, frame_of
from roboflow_workflows.v2.data.masks.grid import MaskGridGeometry, SegmentationView
from roboflow_workflows.v2.data.masks.operations import nearest_full_res
from roboflow_workflows.v2.data.masks.stream_order import StreamOrderedOnce
from roboflow_workflows.v2.data.prediction_ops import RowSelection, take_rows
from roboflow_workflows.v2.data.predictions import (
    INSTANCE_SEGMENTATION_PREDICTION_KIND,
)
from roboflow_workflows.v2.data.rle import check_rle_masks

from inference_models.models.base.instance_segmentation import InstanceDetections
from inference_models.models.base.object_detection import Detections
from inference_models.models.base.types import InstancesRLEMasks
from inference_models.models.common.rle_utils import coco_rle_masks_to_torch_mask

Fidelity = Literal["reference", "approximate"]
FIDELITIES: Tuple[str, ...] = ("reference", "approximate")

BINARY = "binary"
APPROXIMATE_BINARY = "approximate_binary"
SCORE_GRID = "score_grid"

GRID_REFERENCE = "inference_models.align_instance_segmentation_results"
DENSE_REFERENCE = "supplied dense masks"
RLE_REFERENCE = "inference_models.coco_rle_masks_to_torch_mask (pycocotools decode)"


class UnsupportedMaskRequest(ContractError):
    """A provider cannot answer a request without inventing data."""


@dataclass(frozen=True)
class RasterGeometry:
    """The mask raster is the frame raster: no padding, scaling or crop.

    Args:
        size_hw: Raster size ``(H, W)``.
    """

    size_hw: Tuple[int, int]


@dataclass(frozen=True)
class MaskRecipe:
    """What a provider's reference result is, in terms of its producer.

    Args:
        reference: Native producer whose output the reference result equals.
        score_domain: ``logits``, ``probabilities``, ``binary`` or ``coco_rle``.
        score_dtype: Storage dtype of the scores or masks, e.g.
            ``torch.float16`` or ``bytes``.
        threshold: Threshold of ``comparison``; ``None`` when nothing is
            thresholded.
        comparison: When a pixel is foreground, e.g. ``score > threshold``.
        operation_order: The producer's steps, in order, as text.
        geometry: ``MaskGridGeometry`` for a grid, ``RasterGeometry`` otherwise.
    """

    reference: str
    score_domain: str
    score_dtype: str
    threshold: Optional[float]
    comparison: str
    operation_order: Tuple[str, ...]
    geometry: Union[MaskGridGeometry, RasterGeometry]

    @property
    def output_size_hw(self) -> Tuple[int, int]:
        """Size ``(H, W)`` of the dense masks this recipe produces."""
        if isinstance(self.geometry, MaskGridGeometry):
            return tuple(self.geometry.original_size_hw)

        return tuple(self.geometry.size_hw)


@dataclass(frozen=True)
class BinaryMasks:
    """Dense binary masks with the fidelity they actually have.

    Args:
        predictions: ``InstanceDetections`` with a dense ``(N, H, W)`` mask on
            the detections' device. Shared and read-only by contract.
        fidelity: ``reference`` when equal to ``recipe``'s producer output on
            the same arithmetic; ``approximate`` otherwise.
        recipe: The reference these masks equal or approximate.
        method: What was computed, e.g. ``nearest_full_res``.
    """

    predictions: InstanceDetections
    fidelity: Fidelity
    recipe: MaskRecipe
    method: str

    @property
    def frame(self) -> FrameRef:
        """Frame of the masks, read from the predictions' image metadata.

        Raises:
            ContractError: When the metadata has no frame or another size.
        """
        frame = _checked_frame(self.predictions, size_hw=self.recipe.output_size_hw)

        return frame


class _MaskProvider:
    """Shared request handling; subclasses supply rows, recipe and results."""

    capabilities: frozenset = frozenset({BINARY})

    def __len__(self) -> int:
        return len(self.detections.xyxy)

    @property
    def detections(self) -> Detections:
        raise NotImplementedError

    @property
    def recipe(self) -> MaskRecipe:
        raise NotImplementedError

    @property
    def frame(self) -> FrameRef:
        """Frame the masks describe, from the detections' image metadata.

        Raises:
            ContractError: When the metadata has no frame, or the frame's size
                differs from the mask raster.
        """
        frame = _checked_frame(self.detections, size_hw=self.recipe.output_size_hw)

        return frame

    def binary(self, *, fidelity: Fidelity = "reference") -> BinaryMasks:
        """Dense binary masks of every row.

        Args:
            fidelity: ``reference`` requires the reference result.
                ``approximate`` permits a cheaper approximation; a provider
                without one returns its reference, labelled ``reference``.

        Returns:
            The masks, labelled with their actual fidelity and recipe.

        Raises:
            ContractError: On an unknown fidelity, or malformed masks.
        """
        if fidelity not in FIDELITIES:
            raise ContractError(
                f"fidelity must be one of {FIDELITIES}, got {fidelity!r}"
            )

        if fidelity == "approximate" and APPROXIMATE_BINARY in self.capabilities:
            masks = self._approximate()
            return masks

        masks = self._reference()

        return masks

    def score_grid(self) -> Tuple[torch.Tensor, MaskRecipe]:
        """Unthresholded scores with the recipe that thresholds them.

        Raises:
            UnsupportedMaskRequest: The masks carry no scores.
        """
        raise UnsupportedMaskRequest(
            f"{type(self).__name__} holds {self.recipe.score_domain} masks, not "
            "scores; it cannot supply a score grid"
        )

    def with_threshold(self, threshold: float) -> "_MaskProvider":
        """Provider with another threshold; only for scores.

        Raises:
            UnsupportedMaskRequest: The masks carry no scores.
        """
        raise UnsupportedMaskRequest(
            f"{type(self).__name__} holds {self.recipe.score_domain} masks; "
            f"threshold {threshold} cannot be applied without scores"
        )

    def _reference(self) -> BinaryMasks:
        raise NotImplementedError

    def _approximate(self) -> BinaryMasks:
        raise NotImplementedError


class GridMaskProvider(_MaskProvider):
    """Masks as a model's low-resolution grid, in a ``SegmentationView``.

    The reference result is ``view.full_res()``: the model's own dense
    post-processing of the stored scores. ``approximate`` gives
    ``nearest_full_res``: cells thresholded first, then expanded.

    Args:
        view: A model-grid view. Dense-fallback views belong to
            ``DenseMaskProvider``; ``mask_provider_of`` routes them there.

    Raises:
        ContractError: For a dense-fallback view.
    """

    def __init__(self, view: SegmentationView) -> None:
        if not isinstance(view, SegmentationView):
            raise ContractError(
                f"GridMaskProvider wraps a SegmentationView, got {type(view).__name__}"
            )
        if view.is_dense_fallback:
            raise ContractError(
                "A dense-fallback SegmentationView holds dense masks, not a grid; "
                "use DenseMaskProvider or mask_provider_of"
            )

        self._view = view
        self._recipe = _grid_recipe(view)
        self._approximation = StreamOrderedOnce()
        self.capabilities = frozenset({BINARY, APPROXIMATE_BINARY})
        if view.score_type != "binary":
            self.capabilities = self.capabilities | {SCORE_GRID}

    def __len__(self) -> int:
        return len(self._view)

    def __repr__(self) -> str:
        return f"GridMaskProvider({self._view!r})"

    @property
    def view(self) -> SegmentationView:
        """The wrapped view, for its grid operations."""
        return self._view

    @property
    def detections(self) -> Detections:
        """Row-aligned detections. Read-only."""
        return self._view.detections

    @property
    def recipe(self) -> MaskRecipe:
        """The model's dense post-processing of the stored scores."""
        return self._recipe

    def select(
        self,
        *,
        indices: Optional[Sequence] = None,
        mask: Optional[Sequence] = None,
    ) -> "GridMaskProvider":
        """Provider of the selected rows; order and repeats kept.

        Args:
            indices: One-dimensional integer positions.
            mask: One-dimensional boolean selection, one entry per row.

        Returns:
            A provider with its own view and empty caches.

        Raises:
            ContractError: On a malformed selection; see ``RowSelection.of``.
        """
        selection = RowSelection.of(len(self), mask=mask, indices=indices)
        provider = GridMaskProvider(self._view.select(selection))

        return provider

    def with_threshold(self, threshold: float) -> "GridMaskProvider":
        """Provider whose recipe uses another threshold, with empty caches.

        Its reference is the model's post-processing with that threshold, not
        the original one.

        Args:
            threshold: Foreground when ``score > threshold``.

        Returns:
            The provider.

        Raises:
            ContractError: For binary grids; see ``SegmentationView.with_threshold``.
        """
        provider = GridMaskProvider(self._view.with_threshold(threshold))

        return provider

    def score_grid(self) -> Tuple[torch.Tensor, MaskRecipe]:
        """Padded ``(N, mh, mw)`` scores and the recipe that thresholds them.

        Returns:
            The view's scores, read-only, and the recipe.

        Raises:
            UnsupportedMaskRequest: For a binary grid.
        """
        if SCORE_GRID not in self.capabilities:
            raise UnsupportedMaskRequest(
                "this grid holds binary cells, not scores; it cannot supply a "
                "score grid"
            )

        return self._view.scores, self._recipe

    def _reference(self) -> BinaryMasks:
        masks = BinaryMasks(
            predictions=self._view.full_res(),
            fidelity="reference",
            recipe=self._recipe,
            method="SegmentationView.full_res",
        )

        return masks

    def _approximate(self) -> BinaryMasks:
        predictions = self._approximation.get(self._nearest)
        masks = BinaryMasks(
            predictions=predictions,
            fidelity="approximate",
            recipe=self._recipe,
            method="nearest_full_res: threshold cells, nearest-exact expansion",
        )

        return masks

    def _nearest(self) -> InstanceDetections:
        detections = self._view.detections
        predictions = InstanceDetections(
            xyxy=detections.xyxy,
            class_id=detections.class_id,
            confidence=detections.confidence,
            mask=nearest_full_res(self._view),
            image_metadata=detections.image_metadata,
            bboxes_metadata=detections.bboxes_metadata,
        )

        return predictions


class DenseMaskProvider(_MaskProvider):
    """Authoritative dense binary masks, as supplied.

    The supplied raster is the reference: nothing is recomputed and nothing
    is saved. Discarded logits or prompt alternatives cannot be recovered.

    Args:
        predictions: ``InstanceDetections`` with a bool or uint8
            ``(N, H, W)`` mask on the boxes' device. Borrowed, read-only.

    Raises:
        ContractError: For RLE masks or a malformed prediction.
    """

    def __init__(self, predictions: InstanceDetections) -> None:
        INSTANCE_SEGMENTATION_PREDICTION_KIND.check(predictions)
        if not isinstance(predictions.mask, torch.Tensor):
            raise ContractError(
                "DenseMaskProvider needs a dense (N, H, W) mask, got "
                f"{type(predictions.mask).__name__}; use RLEMaskProvider"
            )

        self._predictions = predictions
        self._recipe = MaskRecipe(
            reference=DENSE_REFERENCE,
            score_domain="binary",
            score_dtype=str(predictions.mask.dtype),
            threshold=None,
            comparison="mask value is nonzero",
            operation_order=("supplied by the producer",),
            geometry=RasterGeometry(size_hw=tuple(predictions.mask.shape[1:])),
        )

    def __repr__(self) -> str:
        return (
            f"DenseMaskProvider(rows={len(self)}, "
            f"size_hw={self._recipe.output_size_hw})"
        )

    @property
    def predictions(self) -> InstanceDetections:
        """The supplied predictions, unchanged."""
        return self._predictions

    @property
    def detections(self) -> InstanceDetections:
        """Row-aligned detections: the supplied predictions."""
        return self._predictions

    @property
    def recipe(self) -> MaskRecipe:
        """The supplied raster as its own reference."""
        return self._recipe

    def select(
        self,
        *,
        indices: Optional[Sequence] = None,
        mask: Optional[Sequence] = None,
    ) -> "DenseMaskProvider":
        """Provider of the selected rows; order and repeats kept.

        Args:
            indices: One-dimensional integer positions.
            mask: One-dimensional boolean selection, one entry per row.

        Returns:
            A provider of the selected masks; an identity selection shares
            the mask tensor.

        Raises:
            ContractError: On a malformed selection; see ``RowSelection.of``.
        """
        selection = RowSelection.of(len(self), mask=mask, indices=indices)
        provider = DenseMaskProvider(take_rows(self._predictions, selection))

        return provider

    def _reference(self) -> BinaryMasks:
        masks = BinaryMasks(
            predictions=self._predictions,
            fidelity="reference",
            recipe=self._recipe,
            method="supplied dense masks",
        )

        return masks


class RLEMaskProvider(_MaskProvider):
    """Native COCO RLE masks; decoded only when dense masks are requested.

    Selection keeps the compressed bytes and the recorded image size.
    ``binary()`` decodes the selected rows once, on the CPU, with the native
    helper, then copies the result to the boxes' device.

    Args:
        predictions: ``InstanceDetections`` with ``InstancesRLEMasks``.

    Raises:
        ContractError: For dense masks or a malformed prediction.
    """

    def __init__(self, predictions: InstanceDetections) -> None:
        INSTANCE_SEGMENTATION_PREDICTION_KIND.check(predictions)
        if not isinstance(predictions.mask, InstancesRLEMasks):
            raise ContractError(
                "RLEMaskProvider needs InstancesRLEMasks, got "
                f"{type(predictions.mask).__name__}; use DenseMaskProvider"
            )

        rle = predictions.mask
        self._predictions = predictions
        self._decoded = StreamOrderedOnce()
        self._recipe = MaskRecipe(
            reference=RLE_REFERENCE,
            score_domain="coco_rle",
            score_dtype=_counts_type(rle),
            threshold=None,
            comparison="decoded run value is 1",
            operation_order=(
                "validate compressed counts against image_size",
                "pycocotools decode on the CPU to (H, W, N) uint8",
                "transpose to (N, H, W) bool",
                "copy to the boxes' device when it is not the CPU",
            ),
            geometry=RasterGeometry(size_hw=tuple(rle.image_size)),
        )

    def __repr__(self) -> str:
        return (
            f"RLEMaskProvider(rows={len(self)}, size_hw={self._recipe.output_size_hw}, "
            f"decoded={self._decoded.is_set})"
        )

    @property
    def predictions(self) -> InstanceDetections:
        """The native predictions with their compressed masks, unchanged."""
        return self._predictions

    @property
    def detections(self) -> InstanceDetections:
        """Row-aligned detections: the native predictions."""
        return self._predictions

    @property
    def recipe(self) -> MaskRecipe:
        """The native COCO decode of the stored counts."""
        return self._recipe

    @property
    def is_decoded(self) -> bool:
        """``True`` when ``binary()`` has a cached result."""
        return self._decoded.is_set

    def select(
        self,
        *,
        indices: Optional[Sequence] = None,
        mask: Optional[Sequence] = None,
    ) -> "RLEMaskProvider":
        """Provider of the selected rows; nothing is decoded.

        Args:
            indices: One-dimensional integer positions.
            mask: One-dimensional boolean selection, one entry per row.

        Returns:
            A provider sharing the selected counts objects and image size.

        Raises:
            ContractError: On a malformed selection; see ``RowSelection.of``.
        """
        selection = RowSelection.of(len(self), mask=mask, indices=indices)
        provider = RLEMaskProvider(take_rows(self._predictions, selection))

        return provider

    def _reference(self) -> BinaryMasks:
        masks = BinaryMasks(
            predictions=self._decoded.get(self._decode),
            fidelity="reference",
            recipe=self._recipe,
            method="native COCO RLE decode",
        )

        return masks

    def _decode(self) -> InstanceDetections:
        predictions = self._predictions
        rle = predictions.mask
        check_rle_masks(rle)

        mask = coco_rle_masks_to_torch_mask(rle, device=predictions.xyxy.device)
        decoded = InstanceDetections(
            xyxy=predictions.xyxy,
            class_id=predictions.class_id,
            confidence=predictions.confidence,
            mask=mask,
            image_metadata=predictions.image_metadata,
            bboxes_metadata=predictions.bboxes_metadata,
        )

        return decoded


MaskProvider = Union[GridMaskProvider, DenseMaskProvider, RLEMaskProvider]


def mask_provider_of(payload: Any) -> MaskProvider:
    """Wrap an instance-segmentation payload in its provider, by type only.

    Args:
        payload: A ``SegmentationView`` (a dense fallback unwraps to its dense
            predictions), or ``InstanceDetections`` with a dense or RLE mask.

    Returns:
        The provider of that representation.

    Raises:
        ContractError: For other payloads or malformed predictions.
    """
    if isinstance(payload, SegmentationView):
        if payload.is_dense_fallback:
            provider = DenseMaskProvider(payload.full_res())
            return provider

        provider = GridMaskProvider(payload)
        return provider

    if isinstance(payload, InstanceDetections):
        if isinstance(payload.mask, InstancesRLEMasks):
            provider = RLEMaskProvider(payload)
            return provider

        provider = DenseMaskProvider(payload)
        return provider

    raise ContractError(
        "mask_provider_of supports SegmentationView or InstanceDetections, got "
        f"{type(payload).__name__}"
    )


def _grid_recipe(view: SegmentationView) -> MaskRecipe:
    """The steps ``align_instance_segmentation_results`` takes for this view."""
    geometry = view.geometry
    top, bottom, left, right = geometry.grid_padding_tblr
    crop_x, crop_y = geometry.static_crop_xywh[:2]
    paste = (
        f"paste into the {geometry.original_size_hw} image at ({crop_x}, {crop_y})"
        if crop_x > 0 or crop_y > 0
        else "no paste: the crop starts at (0, 0) and covers the image"
    )
    recipe = MaskRecipe(
        reference=GRID_REFERENCE,
        score_domain=view.score_type,
        score_dtype=str(view.score_dtype),
        threshold=view.threshold,
        comparison="score > threshold",
        operation_order=(
            f"cut grid padding (top, bottom, left, right) = {(top, bottom, left, right)}"
            " cells, rounded; negative padding zero-extends",
            f"bilinear resize in the score dtype to the crop size "
            f"{geometry.crop_size_hw}",
            f"threshold: score > {view.threshold}",
            paste,
        ),
        geometry=geometry,
    )

    return recipe


def _checked_frame(prediction: Any, *, size_hw: Tuple[int, int]) -> FrameRef:
    frame = frame_of(prediction)
    if tuple(frame.size_hw) != tuple(size_hw):
        raise ContractError(
            f"masks are {tuple(size_hw)} (h, w), but the frame in image_metadata "
            f"is {tuple(frame.size_hw)}"
        )

    return frame


def _counts_type(rle: InstancesRLEMasks) -> str:
    kinds = sorted({type(counts).__name__ for counts in rle.masks})

    return " | ".join(kinds) if kinds else "bytes"


__all__ = [
    "APPROXIMATE_BINARY",
    "BINARY",
    "BinaryMasks",
    "DenseMaskProvider",
    "FIDELITIES",
    "Fidelity",
    "GridMaskProvider",
    "MaskProvider",
    "MaskRecipe",
    "RLEMaskProvider",
    "RasterGeometry",
    "SCORE_GRID",
    "UnsupportedMaskRequest",
    "mask_provider_of",
]
