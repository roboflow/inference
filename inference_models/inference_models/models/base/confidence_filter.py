"""
Confidence filter and per-task-type helpers for models with model-eval-derived
`recommendedParameters`.

Concrete models that opt in to recommended_parameters use `ConfidenceFilter`
and the `filter_*` helpers at the end of their `post_process` methods.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch

from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE
from inference_models.weights_providers.entities import RecommendedParameters


class ConfidenceFilter:
    """
    Resolves the 4-tier priority chain (highest to lowest):

      1. Explicit user value — single global threshold for everything
      2. Per-class optimal — per-class filter with the global optimal (or
         hardcoded default) as fallback for classes not in the map
      3. Global optimal — single threshold for everything
      4. Hardcoded default — single threshold for everything

    Exposes:
      - `floor`: the lowest threshold to give the underlying model so its NMS
        doesn't drop boxes we still want to consider for per-class refinement.
      - `passes(class_name, confidence)`: per-detection refinement check.
      - `build_keep_mask(class_ids, confidences, class_names)`: vectorized
        mask for parallel-array detection shapes (OD/IS/KP).
      - `per_class_thresholds(class_names)`: lookup table for shapes that
        index by class_id (e.g. semantic segmentation per-pixel).
      - `has_per_class_refinement`: short-circuit hint when there's nothing
        to refine on top of the floor.
    """

    def __init__(
        self,
        user_confidence: Optional[float] = None,
        recommended_parameters: Optional[RecommendedParameters] = None,
    ):
        # Tier 1: explicit user value wins outright. No per-class refinement
        # needed because the floor IS the final threshold.
        if user_confidence is not None:
            self._floor = user_confidence
            self._per_class: Optional[Dict[str, float]] = None
            self._fallback = user_confidence
            return

        global_optimal = (
            recommended_parameters.confidence
            if recommended_parameters is not None
            else None
        )
        per_class = (
            recommended_parameters.per_class_confidence
            if recommended_parameters is not None
            else None
        )

        # Tier 2: per-class data present.
        if per_class:
            # Classes outside the per-class map fall back to the global
            # optimal, or the hardcoded default if no global was set.
            self._fallback = (
                global_optimal
                if global_optimal is not None
                else INFERENCE_MODELS_DEFAULT_CONFIDENCE
            )
            self._per_class = dict(per_class)
            # Floor must be ≤ every threshold any class might use, so the
            # model doesn't NMS-drop boxes we'd accept after refinement.
            self._floor = min(min(per_class.values()), self._fallback)
            return

        # Tier 3: only global optimal.
        if global_optimal is not None:
            self._floor = global_optimal
            self._per_class = None
            self._fallback = global_optimal
            return

        # Tier 4: hardcoded default.
        self._floor = INFERENCE_MODELS_DEFAULT_CONFIDENCE
        self._per_class = None
        self._fallback = INFERENCE_MODELS_DEFAULT_CONFIDENCE

    @property
    def floor(self) -> float:
        return self._floor

    @property
    def has_per_class_refinement(self) -> bool:
        """True iff `passes` / `build_keep_mask` may return a non-trivial
        result. Used by base classes to short-circuit the per-detection loop
        when there's nothing to refine."""
        return self._per_class is not None

    def passes(self, class_name: str, confidence: float) -> bool:
        """Per-detection refinement check. Returns True for tiers without
        per-class data because the model already filtered at the floor."""
        if not self.has_per_class_refinement:
            return True
        return confidence >= self._per_class.get(class_name, self._fallback)

    def build_keep_mask(
        self,
        class_ids: torch.Tensor,
        confidences: torch.Tensor,
        class_names: List[str],
    ) -> torch.Tensor:
        """Vectorized per-detection mask, for the OD/IS/KP shape where
        `class_ids` and `confidences` are parallel arrays. Caller is
        responsible for short-circuiting via `has_per_class_refinement`
        when applicable; this method also handles the no-refinement case
        (returns all-True)."""
        n = len(class_ids)
        if not self.has_per_class_refinement:
            return torch.ones(n, dtype=torch.bool)

        keep = torch.zeros(n, dtype=torch.bool)
        cid_list = class_ids.tolist()
        conf_list = confidences.tolist()
        for i, (cid, conf) in enumerate(zip(cid_list, conf_list)):
            name = (
                class_names[cid] if 0 <= cid < len(class_names) else str(cid)
            )
            if self.passes(name, conf):
                keep[i] = True
        return keep

    def per_class_thresholds(self, class_names: List[str]) -> List[float]:
        """Return per-class thresholds aligned to `class_names`, for shapes
        that index by class_id (e.g. semantic segmentation per-pixel). For
        tiers without per-class data, every entry is the floor."""
        if not self.has_per_class_refinement:
            return [self._floor] * len(class_names)
        return [
            self._per_class.get(name, self._fallback) for name in class_names
        ]


    # ------------------------------------------------------------------
    # Per-task-type filtering.  Concrete models call these at the end of
    # their post_process when has_per_class_refinement is True.  Imports
    # are deferred to avoid loading task-type modules at import time.
    # ------------------------------------------------------------------

    def filter_detections(self, detections_list, class_names):
        from inference_models.models.base.object_detection import Detections

        result = []
        for detections in detections_list:
            keep = self.build_keep_mask(
                detections.class_id, detections.confidence, class_names
            )
            if bool(keep.all()):
                result.append(detections)
                continue
            bboxes_metadata = detections.bboxes_metadata
            if bboxes_metadata is not None:
                keep_indices = keep.nonzero(as_tuple=True)[0].tolist()
                bboxes_metadata = [bboxes_metadata[i] for i in keep_indices]
            result.append(
                Detections(
                    xyxy=detections.xyxy[keep],
                    class_id=detections.class_id[keep],
                    confidence=detections.confidence[keep],
                    image_metadata=detections.image_metadata,
                    bboxes_metadata=bboxes_metadata,
                )
            )
        return result

    def filter_instance_detections(self, detections_list, class_names):
        from inference_models.models.base.instance_segmentation import (
            InstanceDetections,
        )

        result = []
        for detections in detections_list:
            keep = self.build_keep_mask(
                detections.class_id, detections.confidence, class_names
            )
            if bool(keep.all()):
                result.append(detections)
                continue
            bboxes_metadata = detections.bboxes_metadata
            if bboxes_metadata is not None:
                keep_indices = keep.nonzero(as_tuple=True)[0].tolist()
                bboxes_metadata = [bboxes_metadata[i] for i in keep_indices]
            result.append(
                InstanceDetections(
                    xyxy=detections.xyxy[keep],
                    class_id=detections.class_id[keep],
                    confidence=detections.confidence[keep],
                    mask=detections.mask[keep],
                    image_metadata=detections.image_metadata,
                    bboxes_metadata=bboxes_metadata,
                )
            )
        return result

    def filter_keypoints_and_detections(
        self, keypoints_list, detections_list, class_names
    ):
        from inference_models.models.base.keypoints_detection import KeyPoints
        from inference_models.models.base.object_detection import Detections

        filtered_keypoints = []
        filtered_detections = []
        for kp, det in zip(keypoints_list, detections_list):
            keep = self.build_keep_mask(
                det.class_id, det.confidence, class_names
            )
            if bool(keep.all()):
                filtered_keypoints.append(kp)
                filtered_detections.append(det)
                continue
            filtered_detections.append(
                Detections(
                    xyxy=det.xyxy[keep],
                    class_id=det.class_id[keep],
                    confidence=det.confidence[keep],
                    image_metadata=det.image_metadata,
                    bboxes_metadata=det.bboxes_metadata,
                )
            )
            kp_metadata = kp.key_points_metadata
            if kp_metadata is not None:
                keep_indices = keep.nonzero(as_tuple=True)[0].tolist()
                kp_metadata = [kp_metadata[i] for i in keep_indices]
            filtered_keypoints.append(
                KeyPoints(
                    xy=kp.xy[keep],
                    class_id=kp.class_id[keep],
                    confidence=kp.confidence[keep],
                    image_metadata=kp.image_metadata,
                    key_points_metadata=kp_metadata,
                )
            )
        return filtered_keypoints, filtered_detections

    def filter_multilabel_predictions(self, predictions, class_names):
        from inference_models.models.base.classification import (
            MultiLabelClassificationPrediction,
        )

        result = []
        for prediction in predictions:
            if prediction.class_ids.numel() == 0:
                result.append(prediction)
                continue
            class_ids_list = prediction.class_ids.tolist()
            kept_indices = [
                cid
                for cid in class_ids_list
                if self.passes(
                    class_names[cid]
                    if 0 <= cid < len(class_names)
                    else str(cid),
                    float(prediction.confidence[cid]),
                )
            ]
            if len(kept_indices) == len(class_ids_list):
                result.append(prediction)
                continue
            result.append(
                MultiLabelClassificationPrediction(
                    class_ids=torch.tensor(
                        kept_indices, dtype=prediction.class_ids.dtype
                    ),
                    confidence=prediction.confidence,
                    image_metadata=prediction.image_metadata,
                )
            )
        return result

    def filter_segmentation_results(
        self, results, class_names, background_class_id
    ):
        from inference_models.models.base.semantic_segmentation import (
            SemanticSegmentationResult,
        )

        thresholds = self.per_class_thresholds(class_names)
        filtered = []
        for result in results:
            threshold_tensor = torch.tensor(
                thresholds,
                dtype=result.confidence.dtype,
                device=result.confidence.device,
            )
            per_pixel_thresholds = threshold_tensor[
                result.segmentation_map.long()
            ]
            keep = result.confidence >= per_pixel_thresholds
            if bool(keep.all()):
                filtered.append(result)
                continue
            new_segmentation_map = result.segmentation_map.clone()
            new_confidence = result.confidence.clone()
            new_segmentation_map[~keep] = background_class_id
            new_confidence[~keep] = 0.0
            filtered.append(
                SemanticSegmentationResult(
                    segmentation_map=new_segmentation_map,
                    confidence=new_confidence,
                    image_metadata=result.image_metadata,
                )
            )
        return filtered
