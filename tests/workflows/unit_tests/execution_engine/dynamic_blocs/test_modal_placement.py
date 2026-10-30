"""Placement kwargs the webexec Executor hands to ``app.cls``.

Pinning the executor to one cloud region (aws/us-east-1) meant Modal could
not reschedule preempted containers when that region ran out of capacity, so
the default must stay unpinned to a single cloud region.
"""

import pytest

from .conftest import load_modal_app

_PLACEMENT_ENV_VARS = (
    "WEBEXEC_MODAL_CLOUD",
    "WEBEXEC_MODAL_REGION",
    "WEBEXEC_MODAL_ROUTING_REGION",
)


def _placement_kwargs(monkeypatch, env: dict) -> dict:
    for name in _PLACEMENT_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)

    module = load_modal_app(monkeypatch, "modal_app_placement_under_test")
    kwargs = module._executor_decorator_kwargs
    placement = {
        key: kwargs[key]
        for key in ("cloud", "region", "routing_region")
        if key in kwargs
    }

    return placement


def test_default_placement_is_any_cloud_in_us(monkeypatch) -> None:
    assert _placement_kwargs(monkeypatch, {}) == {"region": "us"}


@pytest.mark.parametrize(
    "env, expected",
    [
        # US deploy matrix entry.
        (
            {
                "WEBEXEC_MODAL_CLOUD": "",
                "WEBEXEC_MODAL_REGION": "us",
                "WEBEXEC_MODAL_ROUTING_REGION": "",
            },
            {"region": "us"},
        ),
        # EU deploy matrix entry keeps its data-residency pin.
        (
            {
                "WEBEXEC_MODAL_CLOUD": "gcp",
                "WEBEXEC_MODAL_REGION": "eu",
                "WEBEXEC_MODAL_ROUTING_REGION": "eu-west",
            },
            {"cloud": "gcp", "region": "eu", "routing_region": "eu-west"},
        ),
        # An explicitly empty region drops the region constraint entirely.
        ({"WEBEXEC_MODAL_REGION": ""}, {}),
    ],
)
def test_placement_follows_deploy_env(monkeypatch, env, expected) -> None:
    assert _placement_kwargs(monkeypatch, env) == expected
