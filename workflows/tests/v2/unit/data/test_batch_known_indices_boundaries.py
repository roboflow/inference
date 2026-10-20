"""``Batch.known_indices`` survives the two engine boundaries that rebuild batches.

resolve_futures   block/phase result  -> Batch(resolved content, same domain)
snapshot_fields   event field values  -> Batch(copied content, same domain)

known_indices   (0, 0) (0, 1) (0, 2) (0, 3)    every known child
indices         (0, 0)        (0, 2)           survivors, what len/iter see
"""

from concurrent.futures import Future
from typing import Any

import numpy as np
from roboflow_workflows.v2.data.batch import Batch
from roboflow_workflows.v2.data.readiness import resolve_futures
from roboflow_workflows.v2.engine.reactions.snapshots import snapshot_fields

KNOWN = ((0, 0), (0, 1), (0, 2), (0, 3))
SURVIVORS = ((0, 0), (0, 2))


def done(value: Any) -> Future:
    future = Future()
    future.set_result(value)

    return future


def survivor_batch(content: list) -> Batch:
    batch = Batch(
        content,
        indices=SURVIVORS,
        known_indices=KNOWN,
        parent_index=(0,),
    )

    return batch


def snapshot(value: Any) -> Any:
    copied = snapshot_fields({"field": value}, where="test").fields["field"]

    return copied


def test_resolved_batch_keeps_survivors_and_the_wider_domain() -> None:
    original = survivor_batch([done(10), done(20)])

    resolved = resolve_futures({"group": original})["group"]

    assert resolved is not original
    assert list(resolved) == [10, 20]
    assert resolved.indices == SURVIVORS
    assert resolved.known_indices == KNOWN
    assert resolved.parent_index == (0,)


def test_resolved_nested_batches_keep_each_domain() -> None:
    inner = Batch(
        [done("a")],
        indices=[(1, 2)],
        known_indices=[(1, 0), (1, 1), (1, 2)],
        parent_index=(1,),
    )
    outer = Batch([survivor_batch([done(10), done(20)]), inner])

    resolved = resolve_futures([outer])[0]

    first, second = list(resolved)
    assert list(first) == [10, 20]
    assert first.known_indices == KNOWN
    assert list(second) == ["a"]
    assert second.indices == ((1, 2),)
    assert second.known_indices == ((1, 0), (1, 1), (1, 2))
    # The outer batch had no wider domain and still has none.
    assert resolved.known_indices is resolved.indices


def test_snapshot_copies_content_and_keeps_the_wider_domain() -> None:
    arrays = [np.zeros(2), np.ones(2)]
    original = survivor_batch(arrays)

    copied = snapshot({"groups": [original]})["groups"][0]

    assert copied is not original
    assert copied[0] is not arrays[0]
    np.testing.assert_array_equal(copied[1], arrays[1])
    assert copied.indices == SURVIVORS
    assert copied.known_indices == KNOWN
    assert copied.parent_index == (0,)


def test_snapshot_keeps_an_all_filtered_group_distinct_from_a_true_empty() -> None:
    all_filtered = Batch([], indices=[], known_indices=[(0, 0)], parent_index=(0,))
    true_empty = Batch.empty(parent_index=(0,))

    copied = snapshot({"filtered": all_filtered, "empty": true_empty})

    assert len(copied["filtered"]) == 0
    assert copied["filtered"].known_indices == ((0, 0),)
    assert copied["empty"].known_indices == ()


def test_batches_without_a_wider_domain_keep_the_default() -> None:
    plain = Batch([done(1), done(2)], indices=[(4, 0), (4, 1)], parent_index=(4,))

    resolved = resolve_futures(plain)
    copied = snapshot(resolved)

    assert resolved.known_indices is resolved.indices
    assert copied.known_indices == ((4, 0), (4, 1))
    assert copied.known_indices == copied.indices
