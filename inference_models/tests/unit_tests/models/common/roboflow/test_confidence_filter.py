"""
Tests for ConfidenceFilter:

  - 4-tier priority chain (user → per-class → global → model default)
  - `refine_*` per-image refinement methods, exercised via minimal stubs that
    mirror the concrete-model pattern of calling ConfidenceFilter inside
    post_process.
"""

from typing import List, Optional, Tuple

import torch

from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE
from inference_models.models.base.classification import (
    MultiLabelClassificationModel,
    MultiLabelClassificationPrediction,
)
from inference_models.models.base.instance_segmentation import (
    InstanceDetections,
    InstanceSegmentationModel,
)
from inference_models.models.base.keypoints_detection import (
    KeyPoints,
    KeyPointsDetectionModel,
)
from inference_models.models.base.object_detection import (
    Detections,
    ObjectDetectionModel,
)
from inference_models.models.base.semantic_segmentation import (
    SemanticSegmentationModel,
    SemanticSegmentationResult,
)
from inference_models.models.common.roboflow.post_processing import ConfidenceFilter
from inference_models.weights_providers.entities import RecommendedParameters


def _rd(*, confidence=None, per_class=None) -> RecommendedParameters:
    return RecommendedParameters(
        confidence=confidence, per_class_confidence=per_class
    )


class TestTier1UserOverride:
    def test_user_value_overrides_everything(self) -> None:
        # Tier 1 short-circuits: per-class and global on recommended_parameters
        # are ignored.
        cf = ConfidenceFilter(
            user_confidence=0.7,
            recommended_parameters=_rd(
                confidence=0.42, per_class={"cat": 0.6, "dog": 0.3}
            ),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == 0.7
        assert cf.has_per_class_refinement is False
        # passes() returns True for everything because the model already
        # filtered at the floor — no per-class refinement in tier 1.
        assert cf.passes("cat", 0.7) is True
        assert cf.passes("dog", 0.7) is True

    def test_explicit_zero_is_honored(self) -> None:
        # Regression: must use `is not None`, not truthiness, so 0.0 doesn't
        # accidentally fall through to per-class.
        cf = ConfidenceFilter(
            user_confidence=0.0,
            recommended_parameters=_rd(confidence=0.42, per_class={"cat": 0.6}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == 0.0

    def test_user_value_honored_when_no_recommended_parameters(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=0.6,
            recommended_parameters=None,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == 0.6
        assert cf.passes("any-class", 0.6) is True


class TestTier2PerClass:
    def test_floor_is_min_of_per_class_and_fallback(self) -> None:
        # Floor must be ≤ every threshold any class might use, so the model's
        # NMS doesn't drop boxes we'd accept after per-class refinement.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(
                confidence=0.5, per_class={"cat": 0.6, "dog": 0.4}
            ),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == 0.4
        assert cf.has_per_class_refinement is True

    def test_passes_checks_per_class_threshold(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(
                confidence=0.5, per_class={"cat": 0.6, "dog": 0.4}
            ),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.passes("cat", 0.6) is True
        assert cf.passes("cat", 0.59) is False
        assert cf.passes("dog", 0.4) is True
        assert cf.passes("dog", 0.39) is False

    def test_unknown_class_falls_back_to_global_optimal(self) -> None:
        # A class not in per_class_confidence (e.g. a class added after eval
        # ran) uses the global optimal as its threshold.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(
                confidence=0.5, per_class={"cat": 0.6, "dog": 0.4}
            ),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.passes("fish", 0.5) is True
        assert cf.passes("fish", 0.49) is False

    def test_unknown_class_falls_back_to_hardcoded_default_when_no_global(self) -> None:
        # Per-class set but no global → unknown classes use the hardcoded default.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(per_class={"cat": 0.6}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == min(0.6, INFERENCE_MODELS_DEFAULT_CONFIDENCE)
        assert cf.passes("dog", INFERENCE_MODELS_DEFAULT_CONFIDENCE) is True
        assert (
            cf.passes("dog", INFERENCE_MODELS_DEFAULT_CONFIDENCE - 0.01) is False
        )


class TestTier3GlobalOnly:
    def test_global_optimal_used_when_no_per_class(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(confidence=0.42),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == 0.42
        assert cf.has_per_class_refinement is False
        assert cf.passes("anything", 0.42) is True

    def test_empty_per_class_treated_as_no_per_class(self) -> None:
        # An empty dict is not "per-class data" — fall through to global optimal.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(confidence=0.42, per_class={}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == 0.42
        assert cf.has_per_class_refinement is False


class TestTier4HardcodedDefault:
    def test_default_used_when_no_recommended_parameters(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=None,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == INFERENCE_MODELS_DEFAULT_CONFIDENCE

    def test_default_used_when_recommended_parameters_is_all_none(self) -> None:
        # Explicit RecommendedParameters with no fields set → still tier 4.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=RecommendedParameters(),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == INFERENCE_MODELS_DEFAULT_CONFIDENCE

    def test_default_constructor_is_no_op(self) -> None:
        # No recommended_parameters or user value → tier 4. Used as a quick
        # "no filter" stand-in.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=None,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.floor == INFERENCE_MODELS_DEFAULT_CONFIDENCE
        assert cf.has_per_class_refinement is False

    def test_custom_default_confidence_used_as_tier_4_floor(self) -> None:
        # When no user value and no recommended_parameters, the model-specific
        # default_confidence arg controls the tier-4 floor.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=None,
            default_confidence=0.25,  # e.g., YOLO26's default
        )
        assert cf.floor == 0.25
        assert cf.passes("any-class", 0.25) is True

    def test_different_default_confidence_values_produce_different_floors(self) -> None:
        # Passing different default_confidence values produces different floor
        # values — this replaces the old mock.patch-based test, since the
        # constant is no longer read inside ConfidenceFilter.
        cf_low = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=None,
            default_confidence=0.1,
        )
        cf_high = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=None,
            default_confidence=0.9,
        )

        assert cf_low.floor == 0.1
        assert cf_high.floor == 0.9


class TestInteractions:
    def test_per_class_takes_precedence_over_global_when_both_set(self) -> None:
        # The 4-tier ordering: with no user value, per-class wins over global.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(confidence=0.5, per_class={"cat": 0.9}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        assert cf.passes("cat", 0.7) is False
        assert cf.passes("cat", 0.9) is True

    def test_filter_is_immutable_against_caller_dict_mutation(self) -> None:
        # Caller's per_class_confidence dict could mutate after construction —
        # the filter takes a copy so it stays stable.
        per_class = {"cat": 0.6}
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(confidence=0.5, per_class=per_class),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        per_class["cat"] = 0.9
        per_class["dog"] = 0.1

        # cat keeps the original threshold (0.6), not the mutated 0.9.
        assert cf.passes("cat", 0.6) is True
        assert cf.passes("cat", 0.59) is False
        # dog was never in the filter's snapshot, so it falls back to the
        # global (0.5), not the mutated 0.1.
        assert cf.passes("dog", 0.5) is True
        assert cf.passes("dog", 0.2) is False


class TestPerClassThresholds:
    """Used by SemanticSegmentationModel to build a per-pixel threshold tensor
    via class_id-indexed lookup. The priority chain itself is exhaustively
    tested via `passes` in the TestTierN classes — these tests just check
    that the indexed-array shape is correct in both tier modes."""

    def test_returns_per_class_thresholds_when_per_class_set(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(
                confidence=0.5, per_class={"cat": 0.6, "dog": 0.4}
            ),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        result = cf.per_class_thresholds(["cat", "dog", "fish"])

        # cat and dog from per_class; fish falls back to global (0.5)
        assert result == [0.6, 0.4, 0.5]

    def test_returns_floor_for_every_class_when_no_per_class_refinement(self) -> None:
        # All non-refinement tiers (1, 3, 4) collapse to the same shape:
        # the floor repeated for every class. Caller can use the same
        # indexed lookup pattern regardless of tier.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=None,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        result = cf.per_class_thresholds(["cat", "dog"])

        assert result == [INFERENCE_MODELS_DEFAULT_CONFIDENCE] * 2


class TestBuildKeepMask:
    """Vectorized mask builder used by the OD/IS/KP base classes for the
    parallel-arrays shape (class_ids[i] aligned with confidences[i])."""

    def test_returns_all_true_when_no_per_class_refinement(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(confidence=0.5),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        mask = cf.build_keep_mask(
            class_ids=torch.tensor([0, 1, 2]),
            confidences=torch.tensor([0.6, 0.4, 0.99]),
            class_names=["cat", "dog", "fish"],
        )

        assert mask.tolist() == [True, True, True]

    def test_filters_per_detection_by_per_class_threshold(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(per_class={"cat": 0.6, "dog": 0.4}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        mask = cf.build_keep_mask(
            class_ids=torch.tensor([0, 0, 1, 1]),
            confidences=torch.tensor([0.7, 0.5, 0.5, 0.3]),
            class_names=["cat", "dog"],
        )

        # cat needs 0.6: 0.7 keeps, 0.5 drops
        # dog needs 0.4: 0.5 keeps, 0.3 drops
        assert mask.tolist() == [True, False, True, False]

    def test_unknown_class_id_falls_back(self) -> None:
        # Out-of-range class_id stringifies and falls through to fallback.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(confidence=0.5, per_class={"cat": 0.9}),
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )

        mask = cf.build_keep_mask(
            class_ids=torch.tensor([99]),
            confidences=torch.tensor([0.6]),
            class_names=["cat"],
        )

        # Class 99 isn't in the map, falls back to global=0.5; 0.6 ≥ 0.5 → keep.
        assert mask.tolist() == [True]

# ---- Stubs that mimic real concrete models: they accept
#      recommended_parameters and use ConfidenceFilter in post_process. ----


class _RecordingObjectDetectionModel(ObjectDetectionModel):
    def __init__(self, detections: Detections):
        self._stub_output = detections
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(detections=None)  # type: ignore[arg-type]

    @property
    def class_names(self) -> List[str]:
        return ["cat", "dog", "fish"]

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        result = [self._stub_output]
        if confidence_filter.has_per_class_refinement:
            result = [
                confidence_filter.refine_detections(r, self.class_names)
                for r in result
            ]
        return result


class _RecordingInstanceSegmentationModel(InstanceSegmentationModel):
    def __init__(self, detections: InstanceDetections):
        self._stub_output = detections
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(detections=None)  # type: ignore[arg-type]

    @property
    def class_names(self) -> List[str]:
        return ["cat", "dog"]

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        result = [self._stub_output]
        if confidence_filter.has_per_class_refinement:
            result = [
                confidence_filter.refine_instance_detections(r, self.class_names)
                for r in result
            ]
        return result


class _RecordingKeypointsDetectionModel(KeyPointsDetectionModel):
    def __init__(self, output: Tuple[List[KeyPoints], List[Detections]]):
        self._stub_output = output
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(output=([], []))  # type: ignore[arg-type]

    @property
    def class_names(self) -> List[str]:
        return ["person", "robot"]

    @property
    def key_points_classes(self) -> List[List[str]]:
        return [["nose"], ["led"]]

    @property
    def skeletons(self):
        return [[], []]

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        kp_list, det_list = self._stub_output
        if confidence_filter.has_per_class_refinement and det_list:
            refined = [
                confidence_filter.refine_keypoints_and_detections(
                    kp, det, self.class_names
                )
                for kp, det in zip(kp_list, det_list)
            ]
            kp_list = [r[0] for r in refined]
            det_list = [r[1] for r in refined]
        return kp_list, det_list


class _RecordingMultiLabelClassificationModel(MultiLabelClassificationModel):
    def __init__(self, predictions: List[MultiLabelClassificationPrediction]):
        self._stub_output = predictions
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(predictions=[])

    @property
    def class_names(self) -> List[str]:
        return ["a", "b", "c", "d"]

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        result = self._stub_output
        if confidence_filter.has_per_class_refinement:
            result = [
                confidence_filter.refine_multilabel_prediction(p, self.class_names)
                for p in result
            ]
        return result


class _RecordingSemanticSegmentationModel(SemanticSegmentationModel):
    _class_names_value = ["cat", "dog", "fish", "background"]
    _background_class_id = 3

    def __init__(self, result: SemanticSegmentationResult):
        self._stub_output = result
        self.recommended_parameters = None
        self.captured_confidence: Optional[float] = None

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        return cls(result=None)  # type: ignore[arg-type]

    @property
    def class_names(self) -> List[str]:
        return self._class_names_value

    def pre_process(self, images, **kwargs):
        raise NotImplementedError

    def forward(self, pre_processed_images, **kwargs):
        raise NotImplementedError

    def post_process(self, model_results, pre_processing_meta, **kwargs):
        confidence_filter = ConfidenceFilter(
            kwargs.get("confidence"),
            self.recommended_parameters,
            default_confidence=INFERENCE_MODELS_DEFAULT_CONFIDENCE,
        )
        self.captured_confidence = confidence_filter.floor
        result = [self._stub_output]
        if confidence_filter.has_per_class_refinement:
            result = [
                confidence_filter.refine_segmentation_result(
                    r, self.class_names, self._background_class_id
                )
                for r in result
            ]
        return result


# ---------------- ObjectDetectionModel ----------------


class TestObjectDetectionPostProcess:
    def _detections(self) -> Detections:
        # Three detections: cat@0.7, dog@0.3, fish@0.5
        return Detections(
            xyxy=torch.tensor(
                [[0, 0, 10, 10], [0, 0, 20, 20], [0, 0, 30, 30]],
                dtype=torch.float32,
            ),
            class_id=torch.tensor([0, 1, 2]),
            confidence=torch.tensor([0.7, 0.3, 0.5]),
        )

    def test_user_value_passes_floor_through_unchanged(self) -> None:
        # Tier 1: explicit user value → floor IS the user value, no per-class
        # refinement, output unchanged.
        model = _RecordingObjectDetectionModel(self._detections())
        model.recommended_parameters = RecommendedParameters(
            confidence=0.5,
            per_class_confidence={"cat": 0.6, "dog": 0.2, "fish": 0.4},
        )

        result = model.post_process(
            model_results=None, pre_processing_meta=None, confidence=0.65
        )

        assert model.captured_confidence == 0.65
        # Tier 1 is no-refinement; output identical to model output.
        assert len(result[0].xyxy) == 3

    def test_per_class_filters_after_floor_is_applied(self) -> None:
        # Tier 2: floor = min(per_class) = 0.2 → model sees 0.2 → all three
        # detections survive NMS → wrapper refines per-class:
        #   cat@0.7 vs cat=0.6 → keep
        #   dog@0.3 vs dog=0.4 → drop
        #   fish@0.5 vs fish=0.45 → keep
        model = _RecordingObjectDetectionModel(self._detections())
        model.recommended_parameters = RecommendedParameters(
            confidence=None,
            per_class_confidence={"cat": 0.6, "dog": 0.4, "fish": 0.45},
        )

        result = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == 0.4  # min(0.6, 0.4, 0.45)
        kept = result[0]
        assert kept.class_id.tolist() == [0, 2]
        torch.testing.assert_close(kept.confidence, torch.tensor([0.7, 0.5]))

    def test_no_per_class_short_circuits(self) -> None:
        # Tier 3: only global. Floor = global. No per-class refinement, output
        # is the same object the model returned.
        original = self._detections()
        model = _RecordingObjectDetectionModel(original)
        model.recommended_parameters = RecommendedParameters(confidence=0.42)

        result = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == 0.42
        assert result[0] is original  # short-circuit returned the original list

    def test_default_when_no_recommended_parameters(self) -> None:
        # Tier 4: no recommended defaults → hardcoded default → no per-class.
        from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE

        model = _RecordingObjectDetectionModel(self._detections())

        model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == INFERENCE_MODELS_DEFAULT_CONFIDENCE


# ---------------- InstanceSegmentationModel ----------------


class TestInstanceSegmentationPostProcess:
    def test_per_class_filter_keeps_mask_aligned(self) -> None:
        # The mask tensor must be sliced in lockstep with class_id/confidence,
        # otherwise filtered detections would carry the wrong masks.
        instance_dets = InstanceDetections(
            xyxy=torch.tensor(
                [[0, 0, 10, 10], [0, 0, 20, 20]], dtype=torch.float32
            ),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([0.7, 0.3]),
            mask=torch.tensor(
                [[[1, 1], [0, 0]], [[0, 0], [1, 1]]], dtype=torch.float32
            ),
        )
        model = _RecordingInstanceSegmentationModel(instance_dets)
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"cat": 0.6, "dog": 0.5}
        )

        result = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        # cat@0.7 ≥ 0.6 keep, dog@0.3 < 0.5 drop
        kept = result[0]
        assert kept.class_id.tolist() == [0]
        # The kept mask must be the cat mask (top row), not the dog mask.
        assert kept.mask.tolist() == [[[1, 1], [0, 0]]]


# ---------------- KeyPointsDetectionModel ----------------


class TestKeypointsDetectionPostProcess:
    def test_per_class_filter_aligns_keypoints_and_detections(self) -> None:
        # The wrapper must filter the parallel KeyPoints and Detections lists
        # with the same boolean mask, so the i-th keypoint set still lines up
        # with the i-th detection after filtering.
        detections = Detections(
            xyxy=torch.tensor(
                [[0, 0, 10, 10], [0, 0, 20, 20]], dtype=torch.float32
            ),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([0.8, 0.2]),
        )
        keypoints = KeyPoints(
            xy=torch.tensor(
                [[[1.0, 2.0]], [[3.0, 4.0]]], dtype=torch.float32
            ),
            class_id=torch.tensor([0, 1]),
            confidence=torch.tensor([[0.9], [0.1]]),
        )
        model = _RecordingKeypointsDetectionModel(
            output=([keypoints], [detections])
        )
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"person": 0.5, "robot": 0.5}
        )

        kp_list, det_list = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        # person@0.8 keep, robot@0.2 drop
        assert det_list[0].class_id.tolist() == [0]
        assert kp_list[0].class_id.tolist() == [0]
        assert kp_list[0].xy.tolist() == [[[1.0, 2.0]]]


# ---------------- MultiLabelClassificationModel ----------------


class TestMultiLabelClassificationPostProcess:
    def test_per_class_filter_drops_class_ids_below_threshold(self) -> None:
        # The full confidence vector stays intact; only the predicted class_ids
        # list is filtered. Per-class threshold uses the indexed score.
        model = _RecordingMultiLabelClassificationModel(
            predictions=[
                MultiLabelClassificationPrediction(
                    class_ids=torch.tensor([0, 1, 2, 3]),
                    confidence=torch.tensor([0.8, 0.4, 0.6, 0.9]),
                )
            ]
        )
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"a": 0.5, "b": 0.5, "c": 0.7, "d": 0.5}
        )

        result = model.post_process(model_results=None)

        kept_ids = result[0].class_ids.tolist()
        # a@0.8 ≥ 0.5 keep, b@0.4 < 0.5 drop, c@0.6 < 0.7 drop, d@0.9 ≥ 0.5 keep
        assert kept_ids == [0, 3]
        # Full confidence vector unchanged.
        torch.testing.assert_close(
            result[0].confidence, torch.tensor([0.8, 0.4, 0.6, 0.9])
        )

    def test_no_per_class_short_circuits_to_model_output(self) -> None:
        original = MultiLabelClassificationPrediction(
            class_ids=torch.tensor([0, 1]),
            confidence=torch.tensor([0.8, 0.7, 0.1, 0.05]),
        )
        model = _RecordingMultiLabelClassificationModel(predictions=[original])
        model.recommended_parameters = RecommendedParameters(confidence=0.5)

        result = model.post_process(model_results=None)

        assert model.captured_confidence == 0.5
        assert result[0] is original


# ---------------- SemanticSegmentationModel ----------------


class TestSemanticSegmentationPostProcess:
    def _result(self) -> SemanticSegmentationResult:
        # 2x2 segmentation map. Pixel layout (class_id, confidence):
        #   (cat, 0.7)  (dog, 0.55)
        #   (fish, 0.5) (cat,  0.3)
        # Background class id = 3 in the stub model.
        return SemanticSegmentationResult(
            segmentation_map=torch.tensor([[0, 1], [2, 0]], dtype=torch.long),
            confidence=torch.tensor(
                [[0.7, 0.55], [0.5, 0.3]], dtype=torch.float32
            ),
        )

    def test_user_value_passes_floor_through_unchanged(self) -> None:
        # Tier 1 — user override beats per-class. The model gets 0.65 as its
        # confidence kwarg, no per-class refinement happens, the result is
        # whatever the model returned (the test stub returns it unmodified).
        model = _RecordingSemanticSegmentationModel(self._result())
        model.recommended_parameters = RecommendedParameters(
            confidence=0.5,
            per_class_confidence={"cat": 0.6, "dog": 0.4, "fish": 0.45},
        )

        out = model.post_process(
            model_results=None, pre_processing_meta=None, confidence=0.65
        )

        assert model.captured_confidence == 0.65
        # No refinement → wrapper passes the model's output through.
        assert out[0] is model._stub_output

    def test_per_class_remaps_below_threshold_pixels_to_background(self) -> None:
        # Tier 2: per_class = {cat: 0.6, dog: 0.4, fish: 0.55}, no global.
        # Floor = min(per_class, hardcoded default). Per-pixel evaluation:
        #   (cat, 0.7)  vs cat=0.6  → keep
        #   (dog, 0.55) vs dog=0.4  → keep
        #   (fish, 0.5) vs fish=0.55 → drop → background (3)
        #   (cat, 0.3)  vs cat=0.6  → drop → background (3)
        model = _RecordingSemanticSegmentationModel(self._result())
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"cat": 0.6, "dog": 0.4, "fish": 0.55},
        )

        out = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert out[0].segmentation_map.tolist() == [[0, 1], [3, 3]]
        # Filtered pixels also get their confidence zeroed for downstream
        # callers that compare confidence vs a threshold.
        torch.testing.assert_close(
            out[0].confidence,
            torch.tensor([[0.7, 0.55], [0.0, 0.0]], dtype=torch.float32),
        )

    def test_no_per_class_short_circuits(self) -> None:
        # Tier 3 — only global. Wrapper passes the model's output through.
        original = self._result()
        model = _RecordingSemanticSegmentationModel(original)
        model.recommended_parameters = RecommendedParameters(confidence=0.42)

        out = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == 0.42
        assert out[0] is original

    def test_default_when_no_recommended_parameters(self) -> None:
        from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE

        model = _RecordingSemanticSegmentationModel(self._result())

        model.post_process(
            model_results=None, pre_processing_meta=None
        )

        assert model.captured_confidence == INFERENCE_MODELS_DEFAULT_CONFIDENCE

    def test_per_class_skipped_when_model_has_no_background_class_id(self) -> None:
        # If a future SS model doesn't expose `_background_class_id`, the
        # wrapper can't safely remap pixels — fall back to the model's
        # floor-only filtering.
        class _NoBackgroundSS(SemanticSegmentationModel):
            # Note: deliberately does NOT set _background_class_id.
            def __init__(self, result):
                self._stub_output = result

            @classmethod
            def from_pretrained(cls, model_name_or_path, **kwargs):
                return cls(result=None)

            @property
            def class_names(self):
                return ["cat", "dog"]

            def pre_process(self, images, **kwargs):
                raise NotImplementedError

            def forward(self, pre_processed_images, **kwargs):
                raise NotImplementedError

            def post_process(self, model_results, pre_processing_meta, **kwargs):
                return [self._stub_output]

        original = SemanticSegmentationResult(
            segmentation_map=torch.tensor([[0, 1]], dtype=torch.long),
            confidence=torch.tensor([[0.7, 0.3]], dtype=torch.float32),
        )
        model = _NoBackgroundSS(original)
        model.recommended_parameters = RecommendedParameters(
            per_class_confidence={"cat": 0.6, "dog": 0.6}
        )

        out = model.post_process(
            model_results=None, pre_processing_meta=None
        )

        # No per-class refinement applied — output is the model's raw result.
        assert out[0] is original
