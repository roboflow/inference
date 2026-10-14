"""Deterministic split of the hosted platform E2E suite into CI shards.

The GitHub Actions workflows run the suite as a matrix of `SHARD_TOTAL` jobs,
each exporting its 0-based `SHARD_INDEX`. Every shard collects the same tests,
sorts their node ids and keeps every `SHARD_TOTAL`-th one starting at
`SHARD_INDEX`. Round-robin over the sorted ids spreads the parametrisations of
slow tests (LLM-backed workflows) across shards instead of piling one test
file onto a single shard. With neither variable set the suite runs whole.
"""

from typing import List, Mapping, Optional, Sequence, Set, Tuple

SHARD_INDEX_ENV = "SHARD_INDEX"
SHARD_TOTAL_ENV = "SHARD_TOTAL"


def resolve_shard(environ: Mapping[str, str]) -> Optional[Tuple[int, int]]:
    """Read the shard selection from environment variables.

    Args:
        environ: Mapping holding the environment, usually `os.environ`.

    Returns:
        `(shard_index, shard_total)`, or None when sharding is not requested.

    Raises:
        ValueError: When only one of the variables is set, a value is not an
            integer, or the index is outside `[0, shard_total)`.
    """
    raw_index = environ.get(SHARD_INDEX_ENV, "").strip()
    raw_total = environ.get(SHARD_TOTAL_ENV, "").strip()
    if not raw_index and not raw_total:
        return None

    if not raw_index or not raw_total:
        raise ValueError(
            f"{SHARD_INDEX_ENV} and {SHARD_TOTAL_ENV} must be set together, "
            f"got {SHARD_INDEX_ENV}='{raw_index}' {SHARD_TOTAL_ENV}='{raw_total}'"
        )

    shard_index = int(raw_index)
    shard_total = int(raw_total)
    if shard_total < 1 or not 0 <= shard_index < shard_total:
        raise ValueError(
            f"Expected 0 <= {SHARD_INDEX_ENV} < {SHARD_TOTAL_ENV}, "
            f"got {SHARD_INDEX_ENV}={shard_index} {SHARD_TOTAL_ENV}={shard_total}"
        )

    return shard_index, shard_total


def select_shard_node_ids(
    node_ids: Sequence[str],
    *,
    shard_index: int,
    shard_total: int,
) -> Set[str]:
    """Pick the node ids that belong to one shard.

    Args:
        node_ids: Node ids of every collected test.
        shard_index: 0-based index of the shard to select.
        shard_total: Number of shards the suite is split into.

    Returns:
        Node ids assigned to the shard. Across all indices the shards are
        disjoint and together cover every input id.
    """
    ordered_node_ids: List[str] = sorted(set(node_ids))
    selected_node_ids = set(ordered_node_ids[shard_index::shard_total])

    return selected_node_ids
