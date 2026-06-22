import pytest
from pydantic import ValidationError

from inference_models.weights_providers.entities import RecommendedParameters


class TestConstruction:
    def test_empty_dict_yields_all_none(self) -> None:
        # Distinct from "no recommended_parameters at all" — the field exists
        # but no values were set yet. All declared fields default to None.
        result = RecommendedParameters.model_validate({})

        assert result.confidence is None
        assert result.per_class_confidence is None


class TestForwardCompat:
    def test_unknown_keys_are_silently_dropped(self) -> None:
        # This is the forward-compat property: if the API adds a new key in a future
        # release, an older inference_models pinned to this version must not crash.
        # The unknown key is dropped; known keys still parse correctly.
        result = RecommendedParameters.model_validate(
            {
                "confidence": 0.42,
                "iouThreshold": 0.7,  # hypothetical future field
                "futureFieldThatDoesNotExistYet": "anything",
            }
        )

        assert result.confidence == 0.42
        assert not hasattr(result, "iou_threshold")
        assert not hasattr(result, "future_field_that_does_not_exist_yet")

    def test_only_unknown_keys_yields_all_none(self) -> None:
        # The whole dict is dropped; all known fields stay at their defaults.
        result = RecommendedParameters.model_validate({"futureFieldOnly": 123})

        assert result.confidence is None


class TestImmutability:
    def test_frozen_blocks_attribute_assignment(self) -> None:
        result = RecommendedParameters(confidence=0.42)

        with pytest.raises(ValidationError):
            result.confidence = 0.5  # type: ignore[misc]


class TestEquality:
    def test_not_equal_when_different_values(self) -> None:
        a = RecommendedParameters(confidence=0.42)
        b = RecommendedParameters(confidence=0.5)

        assert a != b

    def test_construction_equivalence_across_forms(self) -> None:
        # All three construction styles produce equivalent objects, doubling
        # as the proof that all three call shapes work in the first place.
        from_validate = RecommendedParameters.model_validate({"confidence": 0.42})
        from_kwargs = RecommendedParameters(confidence=0.42)
        from_alias = RecommendedParameters(**{"confidence": 0.42})

        assert from_validate == from_kwargs == from_alias


class TestPerClassConfidence:
    def test_parses_from_camel_case_api_payload(self) -> None:
        # Mirror the shape TheGOAT serves: snake_case field aliased to camelCase.
        result = RecommendedParameters.model_validate(
            {"confidence": 0.42, "perClassConfidence": {"cat": 0.45, "dog": 0.4}}
        )

        assert result.confidence == 0.42
        assert result.per_class_confidence == {"cat": 0.45, "dog": 0.4}

    def test_defaults_to_none_when_absent(self) -> None:
        result = RecommendedParameters.model_validate({"confidence": 0.42})

        assert result.per_class_confidence is None

    def test_can_be_set_without_global_confidence(self) -> None:
        # The two fields are independent — per-class alone is also a valid shape.
        result = RecommendedParameters.model_validate(
            {"perClassConfidence": {"cat": 0.5}}
        )

        assert result.confidence is None
        assert result.per_class_confidence == {"cat": 0.5}
