import pytest

from tests.inference.hosted_platform_tests.sharding import (
    resolve_shard,
    select_shard_node_ids,
)


def test_resolve_shard_returns_none_when_not_requested() -> None:
    # when
    result = resolve_shard({})

    # then
    assert result is None


def test_resolve_shard_parses_index_and_total() -> None:
    # when
    result = resolve_shard({"SHARD_INDEX": "2", "SHARD_TOTAL": " 6 "})

    # then
    assert result == (2, 6)


@pytest.mark.parametrize(
    "environ",
    [
        {"SHARD_INDEX": "0"},
        {"SHARD_TOTAL": "6"},
        {"SHARD_INDEX": "6", "SHARD_TOTAL": "6"},
        {"SHARD_INDEX": "-1", "SHARD_TOTAL": "6"},
        {"SHARD_INDEX": "0", "SHARD_TOTAL": "0"},
        {"SHARD_INDEX": "a", "SHARD_TOTAL": "6"},
    ],
)
def test_resolve_shard_rejects_invalid_selection(environ: dict) -> None:
    # when
    with pytest.raises(ValueError):
        _ = resolve_shard(environ)


def test_select_shard_node_ids_partitions_every_test_exactly_once() -> None:
    # given
    node_ids = [f"tests/test_file_{i % 7}.py::test_{i}[legacy]" for i in range(271)]

    # when
    shards = [
        select_shard_node_ids(node_ids, shard_index=index, shard_total=6)
        for index in range(6)
    ]

    # then
    assert set().union(*shards) == set(node_ids)
    assert sum(len(shard) for shard in shards) == len(node_ids)
    assert (
        max(len(shard) for shard in shards) - min(len(shard) for shard in shards) <= 1
    )


def test_select_shard_node_ids_does_not_depend_on_collection_order() -> None:
    # given
    node_ids = [f"tests/test_a.py::test_{i}" for i in range(20)]

    # when
    forward = select_shard_node_ids(node_ids, shard_index=1, shard_total=3)
    backward = select_shard_node_ids(
        list(reversed(node_ids)), shard_index=1, shard_total=3
    )

    # then
    assert forward == backward


def test_select_shard_node_ids_with_single_shard_keeps_everything() -> None:
    # given
    node_ids = ["tests/test_a.py::test_x", "tests/test_b.py::test_y"]

    # when
    result = select_shard_node_ids(node_ids, shard_index=0, shard_total=1)

    # then
    assert result == set(node_ids)
