import pytest

from inference.core.exceptions import FeatureDeprecatedError


def test_feature_deprecated_error_minimum_construction_sets_feature_and_default_message() -> None:
    # when
    error = FeatureDeprecatedError(feature="foo")

    # then
    assert error.feature == "foo"
    assert error.reason is None
    assert error.removal_release is None
    assert error.replacement is None
    assert "Feature 'foo' has been removed from inference." in str(error)
    assert "contact Roboflow" in str(error)


def test_feature_deprecated_error_includes_reason_when_provided() -> None:
    # when
    error = FeatureDeprecatedError(feature="foo", reason="bar")

    # then
    assert error.reason == "bar"
    assert "Reason: bar." in str(error)


def test_feature_deprecated_error_includes_removal_release_when_provided() -> None:
    # when
    error = FeatureDeprecatedError(feature="foo", removal_release="0.99.0")

    # then
    assert error.removal_release == "0.99.0"
    assert "Removed in 0.99.0." in str(error)


def test_feature_deprecated_error_includes_replacement_when_provided() -> None:
    # when
    error = FeatureDeprecatedError(feature="foo", replacement="bar-v2")

    # then
    assert error.replacement == "bar-v2"
    assert "Closest replacement: bar-v2." in str(error)


def test_feature_deprecated_error_get_public_error_details_returns_full_payload() -> None:
    # given
    error = FeatureDeprecatedError(
        feature="foo",
        reason="bar",
        removal_release="0.99.0",
        replacement="bar-v2",
    )

    # when
    details = error.get_public_error_details()

    # then
    assert details == {
        "feature": "foo",
        "removal_release": "0.99.0",
        "replacement": "bar-v2",
        "reason": "bar",
    }


def test_feature_deprecated_error_is_a_python_exception() -> None:
    with pytest.raises(FeatureDeprecatedError):
        raise FeatureDeprecatedError(feature="foo")

    with pytest.raises(Exception):
        raise FeatureDeprecatedError(feature="foo")
