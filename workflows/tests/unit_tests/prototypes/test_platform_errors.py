"""The platform-request exception classes shared with pipeline hosts.

Hosts (the `inference` server and the stream manager's pipeline hosts) raise
and translate these; `inference.core.exceptions` re-exports every one of them,
so names and base chains are client-visible (`error_class`,
`inner_error_type`) and must match the historical server classes exactly.
"""

import ast
import pathlib
import pickle
from typing import List

import pytest
from roboflow_workflows.prototypes import platform_errors

EXPECTED_BASES = {
    "RoboflowAPIRequestError": ["Exception"],
    "RoboflowAPIUnsuccessfulRequestError": ["RoboflowAPIRequestError", "Exception"],
    "RoboflowAPIForbiddenError": [
        "RoboflowAPIUnsuccessfulRequestError",
        "RoboflowAPIRequestError",
        "Exception",
    ],
    "RoboflowAPINotAuthorizedError": [
        "RoboflowAPIUnsuccessfulRequestError",
        "RoboflowAPIRequestError",
        "Exception",
    ],
    "RoboflowAPINotNotFoundError": [
        "RoboflowAPIUnsuccessfulRequestError",
        "RoboflowAPIRequestError",
        "Exception",
    ],
    "RoboflowAPITimeoutError": ["RoboflowAPIRequestError", "Exception"],
    "RoboflowAPIConnectionError": ["RoboflowAPIRequestError", "Exception"],
}


def _base_names(cls: type) -> List[str]:
    return [base.__name__ for base in cls.__mro__[1:-2]]


@pytest.mark.parametrize("name", sorted(EXPECTED_BASES))
def test_names_and_base_chains(name: str) -> None:
    cls = getattr(platform_errors, name)

    assert cls.__name__ == name
    assert cls.__module__ == "roboflow_workflows.prototypes.platform_errors"
    assert _base_names(cls) == EXPECTED_BASES[name]


def test_not_authorized_is_not_forbidden() -> None:
    assert not issubclass(
        platform_errors.RoboflowAPINotAuthorizedError,
        platform_errors.RoboflowAPIForbiddenError,
    )


@pytest.mark.parametrize(
    "name",
    [
        "RoboflowAPINotAuthorizedError",
        "RoboflowAPINotNotFoundError",
        "RoboflowAPITimeoutError",
        "RoboflowAPIConnectionError",
    ],
)
def test_added_classes_keep_plain_exception_behavior(name: str) -> None:
    cls = getattr(platform_errors, name)

    error = cls("request failed")
    restored = pickle.loads(pickle.dumps(error))

    assert str(error) == "request failed"
    assert type(restored) is cls
    assert restored.args == ("request failed",)


def test_the_module_imports_only_typing() -> None:
    source = pathlib.Path(platform_errors.__file__).read_text(encoding="utf-8")
    imported = {
        node.module if isinstance(node, ast.ImportFrom) else alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }

    assert imported == {"typing"}
