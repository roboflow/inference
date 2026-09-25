import numpy as np
import pytest

from inference.core.utils.environment import str2bool as server_str2bool
from inference.core.utils.file_system import (
    ensure_parent_dir_exists as server_ensure_parent_dir_exists,
)
from inference.core.utils.function import experimental as server_experimental
from inference.core.utils.postprocess import cosine_similarity as server_cosine
from inference.core.warnings import InferenceExperimentalFeatureWarning
from inference.core.workflows.errors import WorkflowsInvalidEnvironmentValueError
from inference.core.workflows.utils.text import (
    cosine_similarity,
    ensure_parent_dir_exists,
    experimental,
    str2bool,
)
from inference.core.workflows.warnings import WorkflowsExperimentalFeatureWarning


@pytest.mark.parametrize(
    "value", ["true", "True", "TRUE", "false", "False", True, False]
)
def test_str2bool_matches_the_server_implementation(value) -> None:
    assert str2bool(value) == server_str2bool(value)


def test_str2bool_rejects_non_boolean_spellings() -> None:
    # BOTH implementations raise here - the original plan asserted equality on
    # "1" and "0", which cannot pass because neither returns a value for them.
    # Only the exception TYPE differs, and that difference is the point of the
    # workflows-local class.
    with pytest.raises(WorkflowsInvalidEnvironmentValueError):
        str2bool("1")


def test_cosine_similarity_matches_the_server_implementation() -> None:
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert cosine_similarity(a, b) == server_cosine(a, b)


def test_experimental_matches_the_server_implementation() -> None:
    reason = "pinned by this test"

    def some_function() -> int:
        return 7

    workflows_decorated = experimental(reason=reason)(some_function)

    def some_function() -> int:
        return 7

    server_decorated = server_experimental(reason=reason)(some_function)

    with pytest.warns(WorkflowsExperimentalFeatureWarning) as workflows_record:
        assert workflows_decorated() == 7
    with pytest.warns(InferenceExperimentalFeatureWarning) as server_record:
        assert server_decorated() == 7

    assert str(workflows_record[0].message) == str(server_record[0].message)


def test_ensure_parent_dir_exists_creates_missing_parents(tmp_path) -> None:
    target = tmp_path / "a" / "b" / "file.txt"
    server_target = tmp_path / "c" / "d" / "file.txt"

    ensure_parent_dir_exists(str(target))
    server_ensure_parent_dir_exists(str(server_target))

    assert target.parent.is_dir()
    assert server_target.parent.is_dir()
