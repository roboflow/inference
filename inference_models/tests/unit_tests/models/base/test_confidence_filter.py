"""
Tests for ConfidenceFilter — built once per inference call from
(user_confidence, recommended_parameters). Resolves the 4-tier priority chain
(user → per-class → global → hardcoded default) and is what the base class
wrappers use during postprocess.
"""

from unittest import mock

import torch

from inference_models.configuration import INFERENCE_MODELS_DEFAULT_CONFIDENCE
from inference_models.models.base.confidence_filter import ConfidenceFilter
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
        )

        assert cf.floor == 0.0

    def test_user_value_honored_when_no_recommended_parameters(self) -> None:
        cf = ConfidenceFilter(user_confidence=0.6, recommended_parameters=None)

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
        )

        assert cf.floor == 0.4
        assert cf.has_per_class_refinement is True

    def test_passes_checks_per_class_threshold(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(
                confidence=0.5, per_class={"cat": 0.6, "dog": 0.4}
            ),
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
        )

        assert cf.passes("fish", 0.5) is True
        assert cf.passes("fish", 0.49) is False

    def test_unknown_class_falls_back_to_hardcoded_default_when_no_global(self) -> None:
        # Per-class set but no global → unknown classes use the hardcoded default.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(per_class={"cat": 0.6}),
        )

        assert cf.floor == min(0.6, INFERENCE_MODELS_DEFAULT_CONFIDENCE)
        assert cf.passes("dog", INFERENCE_MODELS_DEFAULT_CONFIDENCE) is True
        assert (
            cf.passes("dog", INFERENCE_MODELS_DEFAULT_CONFIDENCE - 0.01) is False
        )


class TestTier3GlobalOnly:
    def test_global_optimal_used_when_no_per_class(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=None, recommended_parameters=_rd(confidence=0.42)
        )

        assert cf.floor == 0.42
        assert cf.has_per_class_refinement is False
        assert cf.passes("anything", 0.42) is True

    def test_empty_per_class_treated_as_no_per_class(self) -> None:
        # An empty dict is not "per-class data" — fall through to global optimal.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(confidence=0.42, per_class={}),
        )

        assert cf.floor == 0.42
        assert cf.has_per_class_refinement is False


class TestTier4HardcodedDefault:
    def test_default_used_when_no_recommended_parameters(self) -> None:
        cf = ConfidenceFilter(user_confidence=None, recommended_parameters=None)

        assert cf.floor == INFERENCE_MODELS_DEFAULT_CONFIDENCE

    def test_default_used_when_recommended_parameters_is_all_none(self) -> None:
        # Explicit RecommendedParameters with no fields set → still tier 4.
        cf = ConfidenceFilter(
            user_confidence=None, recommended_parameters=RecommendedParameters()
        )

        assert cf.floor == INFERENCE_MODELS_DEFAULT_CONFIDENCE

    def test_default_constructor_is_no_op(self) -> None:
        # No args at all → tier 4. Used as a quick "no filter" stand-in.
        cf = ConfidenceFilter()

        assert cf.floor == INFERENCE_MODELS_DEFAULT_CONFIDENCE
        assert cf.has_per_class_refinement is False

    @mock.patch(
        "inference_models.models.base.confidence_filter.INFERENCE_MODELS_DEFAULT_CONFIDENCE",
        0.123,
    )
    def test_fallback_reads_current_value_of_default_constant(self) -> None:
        # Env-var overrides at startup change the constant — the filter must
        # read it at call time, not bake it in at import.
        cf = ConfidenceFilter(user_confidence=None, recommended_parameters=None)

        assert cf.floor == 0.123


class TestInteractions:
    def test_per_class_takes_precedence_over_global_when_both_set(self) -> None:
        # The 4-tier ordering: with no user value, per-class wins over global.
        cf = ConfidenceFilter(
            user_confidence=None,
            recommended_parameters=_rd(confidence=0.5, per_class={"cat": 0.9}),
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
        )

        result = cf.per_class_thresholds(["cat", "dog", "fish"])

        # cat and dog from per_class; fish falls back to global (0.5)
        assert result == [0.6, 0.4, 0.5]

    def test_returns_floor_for_every_class_when_no_per_class_refinement(self) -> None:
        # All non-refinement tiers (1, 3, 4) collapse to the same shape:
        # the floor repeated for every class. Caller can use the same
        # indexed lookup pattern regardless of tier.
        cf = ConfidenceFilter()

        result = cf.per_class_thresholds(["cat", "dog"])

        assert result == [INFERENCE_MODELS_DEFAULT_CONFIDENCE] * 2


class TestBuildKeepMask:
    """Vectorized mask builder used by the OD/IS/KP base classes for the
    parallel-arrays shape (class_ids[i] aligned with confidences[i])."""

    def test_returns_all_true_when_no_per_class_refinement(self) -> None:
        cf = ConfidenceFilter(
            user_confidence=None, recommended_parameters=_rd(confidence=0.5)
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
        )

        mask = cf.build_keep_mask(
            class_ids=torch.tensor([99]),
            confidences=torch.tensor([0.6]),
            class_names=["cat"],
        )

        # Class 99 isn't in the map, falls back to global=0.5; 0.6 ≥ 0.5 → keep.
        assert mask.tolist() == [True]
