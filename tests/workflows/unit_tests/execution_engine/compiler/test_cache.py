import pytest

from inference.core.workflows.errors import WorkflowEnvironmentConfigurationError
from inference.core.workflows.execution_engine.v1.compiler.cache import (
    BasicWorkflowsCache,
)


def test_cache_when_hash_key_cannot_be_obtained() -> None:
    # given
    cache = BasicWorkflowsCache[int](
        cache_size=16,
        hash_functions=[("some", lambda v: str(v))],
    )

    # when
    with pytest.raises(WorkflowEnvironmentConfigurationError):
        _ = cache.get_hash_key(other=3)


def test_cache_when_hash_key_can_be_obtained() -> None:
    # given
    cache = BasicWorkflowsCache[int](
        cache_size=16,
        hash_functions=[("some", lambda v: str(v))],
    )

    # when
    value = cache.get_hash_key(some=3)

    # then
    assert isinstance(value, str)


def test_get_cache_value_on_cache_hit() -> None:
    # given
    cache = BasicWorkflowsCache[str](
        cache_size=16,
        hash_functions=[("some", lambda v: str(v))],
    )
    key = cache.get_hash_key(some=3)
    cache.cache(key=key, value="my_value")

    # when
    result = cache.get(key=key)

    # then
    assert result == "my_value"


def test_get_cache_value_on_cache_miss() -> None:
    # given
    cache = BasicWorkflowsCache[str](
        cache_size=16,
        hash_functions=[("some", lambda v: str(v))],
    )

    # when
    result = cache.get(key="my_key")

    # then
    assert result is None


def test_cache_being_emptied_properly() -> None:
    # given
    cache = BasicWorkflowsCache[str](
        cache_size=2,
        hash_functions=[("some", lambda v: str(v))],
    )

    # when
    key_one = cache.get_hash_key(some=1)
    cache.cache(key=key_one, value="my_value_1")

    key_two = cache.get_hash_key(some=2)
    cache.cache(key=key_two, value="my_value_2")

    key_three = cache.get_hash_key(some=3)
    cache.cache(key=key_three, value="my_value_3")

    # then
    assert cache.get(key_one) is None
    assert cache.get(key_two) == "my_value_2"
    assert cache.get(key_three) == "my_value_3"
