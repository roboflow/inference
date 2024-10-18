import pytest
from pydantic import ValidationError

from inference.core.workflows.core_steps.flow_control.rate_limiter.v1 import (
    RateLimiterManifest,
)


@pytest.mark.parametrize(
    "depends_on_selector", ["$inputs.image", "$inputs.param", "$steps.some.data"]
)
def test_rate_limiter_manifest_parsing_when_input_is_valid(
    depends_on_selector: str,
) -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/rate_limiter@v1",
        "name": "rate_limiter",
        "cooldown_seconds": 10,
        "depends_on": depends_on_selector,
        "next_steps": ["$steps.some"],
    }

    # when
    result = RateLimiterManifest.model_validate(raw_manifest)

    # then
    assert result == RateLimiterManifest(
        type="roboflow_core/rate_limiter@v1",
        name="rate_limiter",
        cooldown_seconds=10,
        depends_on=depends_on_selector,
        next_steps=["$steps.some"],
    )


def test_rate_limiter_manifest_parsing_when_cooldown_is() -> None:
    # given
    raw_manifest = {
        "type": "roboflow_core/rate_limiter@v1",
        "name": "rate_limiter",
        "cooldown_seconds": -0.1,
        "depends_on": "$steps.some.data",
        "next_steps": ["$steps.some"],
    }

    # when
    with pytest.raises(ValidationError):
        _ = RateLimiterManifest.model_validate(raw_manifest)
