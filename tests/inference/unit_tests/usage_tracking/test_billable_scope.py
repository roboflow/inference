"""Tests for the request-scoped billing intent and how it crosses to a remote server.

The context variable covers work done in-process. A Workflow block running in
``StepExecutionMode.REMOTE`` records nothing locally - the server it calls runs
the model and bills it - so the scope has to be rendered into request
parameters. These tests pin both halves, plus the fact that every block with a
remote path actually forwards them.
"""

import ast
import pathlib

import pytest

from inference.usage_tracking import billable_scope
from inference.usage_tracking.billable_scope import (
    billing_suppressed,
    remote_billing_parameters,
)

CORE_STEPS_ROOT = pathlib.Path(billable_scope.__file__).parents[1] / (
    "core/workflows/core_steps"
)
HELPER_NAME = "remote_billing_parameters"


def test_no_parameters_are_forwarded_outside_a_suppressed_scope(
    configured_service_secret,
):
    assert remote_billing_parameters() == {}


def test_parameters_carry_the_opt_out_and_the_secret(configured_service_secret):
    with billing_suppressed(True):
        assert remote_billing_parameters() == {
            "count_inference": False,
            "service_secret": configured_service_secret,
        }


def test_parameters_do_not_outlive_the_scope(configured_service_secret):
    with billing_suppressed(True):
        pass

    assert remote_billing_parameters() == {}


def test_a_suppressed_scope_forwards_nothing_without_a_configured_secret(monkeypatch):
    # The receiving server re-validates, so an unprovable opt-out is not worth
    # sending - and must never downgrade billing on its own.
    monkeypatch.setattr(billable_scope, "ROBOFLOW_SERVICE_SECRET", None)

    with billing_suppressed(True):
        assert remote_billing_parameters() == {}


def test_an_explicitly_unsuppressed_scope_forwards_nothing(configured_service_secret):
    with billing_suppressed(False):
        assert remote_billing_parameters() == {}


def _configuration_calls(tree: ast.AST):
    return [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "InferenceConfiguration"
    ]


def _spreads_billing_parameters(call: ast.Call) -> bool:
    return any(
        keyword.arg is None
        and isinstance(keyword.value, ast.Call)
        and isinstance(keyword.value.func, ast.Name)
        and keyword.value.func.id == HELPER_NAME
        for keyword in call.keywords
    )


def _blocks_building_a_remote_client():
    for path in sorted(CORE_STEPS_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for call in _configuration_calls(tree):
            yield path, call


def test_every_block_building_a_remote_client_forwards_the_billing_scope():
    # A new block that builds its own InferenceConfiguration silently bills the
    # caller who opted out, and nothing else would catch it.
    offenders = [
        str(path.relative_to(CORE_STEPS_ROOT))
        for path, call in _blocks_building_a_remote_client()
        if not _spreads_billing_parameters(call)
    ]

    assert not offenders, (
        "these blocks build an InferenceConfiguration without spreading "
        f"**{HELPER_NAME}(), so an authenticated countinference=false never "
        f"reaches the server that bills the model: {offenders}"
    )


def test_the_scan_actually_inspects_blocks():
    # Without this the guard above would pass vacuously if the glob broke.
    assert len(list(_blocks_building_a_remote_client())) > 50


def test_the_scan_rejects_a_configuration_without_the_spread():
    tree = ast.parse("InferenceConfiguration(api_key_transport='both')")

    assert not _spreads_billing_parameters(_configuration_calls(tree)[0])


@pytest.mark.parametrize(
    "source",
    [
        f"InferenceConfiguration(**{HELPER_NAME}())",
        f"InferenceConfiguration(api_key_transport='both', **{HELPER_NAME}())",
    ],
)
def test_the_scan_accepts_a_configuration_with_the_spread(source):
    tree = ast.parse(source)

    assert _spreads_billing_parameters(_configuration_calls(tree)[0])
