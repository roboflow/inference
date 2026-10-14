from dataclasses import dataclass
from math import isfinite
from numbers import Integral, Real
from typing import List, Literal, Optional, Tuple, TypeVar

import torch

PreprocessedInputs = TypeVar("PreprocessedInputs")
PreprocessingMetadata = TypeVar("PreprocessingMetadata")
RawPrediction = TypeVar("RawPrediction")


@dataclass
class InstancesRLEMasks:
    image_size: Tuple[int, int]  # (h, w) of the image the masks describe
    masks: List[bytes]
    # (h, w) of the grid the masks are encoded on. Defaults to image_size,
    # which is the only value it took before the grid became adjustable.
    mask_size: Optional[Tuple[int, int]] = None

    def __post_init__(self) -> None:
        if self.mask_size is None:
            self.mask_size = self.image_size

    @classmethod
    def from_coco_rle_masks(
        cls,
        image_size: Tuple[int, int],
        masks: List[dict],
        mask_size: Optional[Tuple[int, int]] = None,
    ) -> "InstancesRLEMasks":
        masks = [m["counts"] for m in masks]
        return cls(image_size=image_size, masks=masks, mask_size=mask_size)

    def to_coco_rle_masks(self) -> List[dict]:
        # COCO `size` describes the grid the counts were encoded on. When that
        # is not the image grid, declaring the image size would make the counts
        # disagree with it, which pycocotools decodes silently wrong.
        encoded_size = list(self.mask_size or self.image_size)

        return [{"size": encoded_size, "counts": m} for m in self.masks]


@dataclass(frozen=True)
class InstanceSegmentationGrid:
    """Selected native mask scores with resolved image-placement geometry.

    Scores already include the producer's model-specific combination, activation
    and box crop, where applicable. The supported reconstruction is signed grid
    unpadding, bilinear score resize, strict thresholding, then binary placement
    on an image-representing canvas. This is not a carrier for raw prototypes or
    arbitrary model reconstruction recipes.

    The enclosing ``InstanceDetections`` uses original-image logical coordinates
    in all three size fields and unrounded, unclipped float32 image-space boxes.
    ``scores.shape[-2:]`` describes separate physical storage, not ``mask_size``.
    The carrier owns no detection fields, resolution preference or CUDA lifecycle.
    Treat its tensors as read-only; frozen dataclasses do not freeze tensor storage.
    Producers must publish independent snapshots safe from later buffer reuse.

    Args:
        scores: Floating ``(N, mh, mw)`` grids in selected detection order.
        score_type: Whether scores are logits or probabilities, after preparation.
        threshold: Finite foreground threshold, using strict ``>`` after resize.
        grid_padding_ltrb: Signed rounded grid-cell padding; negative extends zeros.
        image_crop_xywh: Destination crop in original-image pixels at factor one.

    Raises:
        ValueError: If shape, score domain, threshold or scalar geometry is invalid.
    """

    scores: torch.Tensor
    score_type: Literal["logits", "probabilities"]
    threshold: float
    grid_padding_ltrb: Tuple[int, int, int, int]
    image_crop_xywh: Tuple[int, int, int, int]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.scores, torch.Tensor)
            or self.scores.ndim != 3
            or not self.scores.dtype.is_floating_point
            or min(self.scores.shape[1:]) <= 0
        ):
            raise ValueError(
                "scores must be floating (N, mh, mw) with positive grid sizes"
            )
        if self.score_type not in ("logits", "probabilities"):
            raise ValueError("score_type must be logits or probabilities")
        if (
            isinstance(self.threshold, bool)
            or not isinstance(self.threshold, Real)
            or not isfinite(self.threshold)
        ):
            raise ValueError("threshold must be finite")
        if self.score_type == "probabilities" and not 0 <= self.threshold <= 1:
            raise ValueError("probability threshold must be in [0, 1]")

        for name in ("grid_padding_ltrb", "image_crop_xywh"):
            values = getattr(self, name)
            if (
                not isinstance(values, tuple)
                or len(values) != 4
                or any(
                    isinstance(v, bool) or not isinstance(v, Integral) for v in values
                )
            ):
                raise ValueError(f"{name} must be a tuple of four integers")

        left, top, right, bottom = self.grid_padding_ltrb
        height, width = self.scores.shape[1:]
        if height - top - bottom <= 0 or width - left - right <= 0:
            raise ValueError("grid padding must leave a nonempty score grid")
        x, y, crop_width, crop_height = self.image_crop_xywh
        if min(x, y) < 0 or min(crop_width, crop_height) <= 0:
            raise ValueError(
                "image crop must have nonnegative origin and positive size"
            )
