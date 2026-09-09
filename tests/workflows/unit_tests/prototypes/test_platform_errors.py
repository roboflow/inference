"""The four Roboflow-API exception classes live in workflows and the server
re-exports them - one class object, so nothing observable changes.

`post_to_roboflow_api` raises these (401 -> RoboflowAPINotAuthorizedError, a
subclass of RoboflowAPIUnsuccessfulRequestError; 403 -> RoboflowAPIForbiddenError;
any other 4xx/5xx -> RoboflowAPIUnsuccessfulRequestError), and blocks catch
them - so identity, not just name, has to survive.
"""

import ast
import pathlib

import pytest

from inference.core import exceptions as server_exceptions
from inference.core.interfaces.workflows_step_error_handlers import (
    extended_roboflow_errors_handler,
    legacy_step_error_handler,
)
from inference.core.workflows.errors import ClientCausedStepExecutionError
from inference.core.workflows.prototypes import platform_errors

RELOCATED = [
    "RoboflowAPIRequestError",
    "RoboflowAPIUnsuccessfulRequestError",
    "RoboflowAPIForbiddenError",
    "FeatureDeprecatedError",
]


@pytest.mark.parametrize("name", RELOCATED)
def test_the_server_name_is_the_workflows_class(name) -> None:
    assert getattr(server_exceptions, name) is getattr(platform_errors, name)


@pytest.mark.parametrize("name", RELOCATED)
def test_the_public_class_name_is_unchanged(name) -> None:
    # `WorkflowError.inner_error_type` and `error_handlers.py` serialize
    # `__class__.__name__`; a different name is a wire change.
    assert getattr(platform_errors, name).__name__ == name


def test_the_subclass_hierarchy_is_unchanged() -> None:
    E = server_exceptions
    assert issubclass(E.RoboflowAPIUnsuccessfulRequestError, E.RoboflowAPIRequestError)
    assert issubclass(
        E.RoboflowAPINotAuthorizedError, E.RoboflowAPIUnsuccessfulRequestError
    )
    assert issubclass(
        E.RoboflowAPIForbiddenError, E.RoboflowAPIUnsuccessfulRequestError
    )
    assert issubclass(E.PaymentRequiredError, E.RoboflowAPIUnsuccessfulRequestError)
    assert issubclass(
        E.RoboflowAPIUsagePausedError, E.RoboflowAPIUnsuccessfulRequestError
    )
    assert issubclass(
        E.RoboflowAPINotNotFoundError, E.RoboflowAPIUnsuccessfulRequestError
    )
    assert issubclass(E.ModelManagerLockAcquisitionError, E.RoboflowAPIRequestError)
    assert issubclass(E.RoboflowAPIConnectionError, E.RoboflowAPIRequestError)
    assert issubclass(E.RoboflowAPITimeoutError, E.RoboflowAPIRequestError)


def test_the_deprecation_constructor_and_details_are_unchanged() -> None:
    error = server_exceptions.FeatureDeprecatedError(
        feature="roboflow_core/cog_vlm@v1",
        reason="End-of-life due to CVE-2024-11393",
        removal_release="0.54.0",
    )
    assert error.get_structured_public_error_details() == {
        "feature": "roboflow_core/cog_vlm@v1",
        "removal_release": "0.54.0",
        "replacement": None,
        "reason": "End-of-life due to CVE-2024-11393",
    }
    assert "roboflow_core/cog_vlm@v1" in str(error)
    assert "Reason: End-of-life due to CVE-2024-11393" in str(error)


def test_the_module_imports_only_typing() -> None:
    path = (
        pathlib.Path(__file__).resolve().parents[4]
        / "inference/core/workflows/prototypes/platform_errors.py"
    )
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names)
    assert modules <= {"typing"}, (
        f"platform_errors.py must stay dependency-free ({modules}) - "
        "exceptions.py imports it"
    )


@pytest.mark.parametrize(
    "handler", [legacy_step_error_handler, extended_roboflow_errors_handler]
)
def test_deprecation_still_maps_to_410_from_the_three_real_raise_sites(handler) -> None:
    for feature, reason in (
        ("roboflow_core/cog_vlm@v1", "End-of-life due to CVE-2024-11393"),
        ("roboflow_core/gaze@v1", "MediaPipe dependency removed from inference"),
        ("roboflow_core/yolo_world_model@v1", "YOLO-World is deprecated"),
    ):
        error = server_exceptions.FeatureDeprecatedError(feature=feature, reason=reason)
        with pytest.raises(ClientCausedStepExecutionError) as raised:
            handler("some_step", error)
        assert raised.value.status_code == 410
        assert raised.value.inner_error_type == "FeatureDeprecatedError"


def test_a_proxy_403_still_maps_to_403_with_the_original_error_name() -> None:
    """End to end through the block's own callback: the serialized
    `inner_error_type` must stay `RoboflowAPIForbiddenError`."""
    import requests

    from inference.core.workflows.core_steps.common import openrouter

    response = requests.Response()
    response.status_code = 403
    response._content = b'{"details": "not allowed"}'
    http_error = requests.exceptions.HTTPError(response=response)

    with pytest.raises(server_exceptions.RoboflowAPIForbiddenError) as raised:
        openrouter._PROXY_ERROR_HANDLERS[403](http_error)
    assert raised.value.status_code == 403

    with pytest.raises(ClientCausedStepExecutionError) as mapped:
        extended_roboflow_errors_handler("some_step", raised.value)
    assert mapped.value.status_code == 403
    assert mapped.value.inner_error_type == "RoboflowAPIForbiddenError"


def test_a_401_still_maps_to_401() -> None:
    with pytest.raises(ClientCausedStepExecutionError) as error:
        extended_roboflow_errors_handler(
            "some_step", server_exceptions.RoboflowAPINotAuthorizedError("nope")
        )
    assert error.value.status_code == 401


def test_a_generic_unsuccessful_request_is_still_unmapped() -> None:
    assert (
        extended_roboflow_errors_handler(
            "some_step", server_exceptions.RoboflowAPIUnsuccessfulRequestError("boom")
        )
        is None
    )


def test_no_workflows_module_imports_the_server_exceptions() -> None:
    workflows_root = (
        pathlib.Path(__file__).resolve().parents[4] / "inference" / "core" / "workflows"
    )
    offenders = []
    for path in workflows_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "inference.core.exceptions"
            ):
                offenders.append(f"{path}:{node.lineno}")
    assert not offenders, offenders
