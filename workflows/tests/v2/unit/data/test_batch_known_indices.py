"""``Batch.known_indices``: the grouping domain beside the surviving indices.

known_indices   (0, 0) (0, 1) (0, 2) (0, 3)    every known child
indices         (0, 0)        (0, 2)           survivors, what len/iter see
"""

import pytest
from roboflow_workflows.v2.data.batch import Axis, Batch, EntryLayout, EntryMetadata
from roboflow_workflows.v2.engine.errors import ContractError

KNOWN = [(0, 0), (0, 1), (0, 2), (0, 3)]


def survivor_batch() -> Batch:
    batch = Batch(
        [10, 20],
        indices=[(0, 0), (0, 2)],
        known_indices=KNOWN,
        parent_index=(0,),
    )

    return batch


def test_known_indices_default_to_indices_without_new_metadata() -> None:
    explicit = Batch(["a", "b"], indices=[(3, 1), (3, 4)], parent_index=(3,))
    local = Batch.of(["a", "b"])
    empty = Batch.empty(parent_index=(2,))

    assert explicit.known_indices is explicit.indices
    assert local.known_indices == ((0,), (1,))
    assert empty.known_indices == ()


def test_survivor_view_is_unchanged_by_a_wider_domain() -> None:
    batch = survivor_batch()

    assert batch.known_indices == tuple(KNOWN)
    assert batch.indices == ((0, 0), (0, 2))
    assert len(batch) == 2
    assert list(batch) == [10, 20]
    assert batch[1] == 20
    assert list(batch.iter_with_indices()) == [((0, 0), 10), ((0, 2), 20)]
    # Equality and repr stay survivor-based, so existing comparisons hold.
    assert batch == Batch([10, 20], indices=[(0, 0), (0, 2)], parent_index=(0,))
    assert "known" not in repr(batch)


def test_all_filtered_domain_differs_from_a_genuinely_empty_group() -> None:
    all_filtered = Batch([], indices=[], known_indices=[(1, 0)], parent_index=(1,))
    genuinely_empty = Batch.empty(parent_index=(1,))

    assert len(all_filtered) == len(genuinely_empty) == 0
    assert all_filtered.known_indices == ((1, 0),)
    assert genuinely_empty.known_indices == ()


def test_known_indices_are_read_only() -> None:
    batch = survivor_batch()

    with pytest.raises(AttributeError):
        batch.known_indices = ()
    with pytest.raises(AttributeError):
        batch._extra = 1


def test_known_indices_accept_any_iterable_once() -> None:
    batch = Batch(
        [10],
        indices=[(0, 2)],
        known_indices=(index for index in KNOWN),
        parent_index=(0,),
    )

    assert batch.known_indices == tuple(KNOWN)


def test_with_view_preserves_the_known_domain() -> None:
    batch = survivor_batch()
    layout = EntryLayout(axes=(Axis(id="crops", kind="dynamic_nesting"),))
    metadata = EntryMetadata()

    viewed = batch.with_view(layout=layout, metadata=metadata)

    assert viewed is not batch
    assert viewed.layout is layout
    assert viewed.known_indices == tuple(KNOWN)
    assert viewed.indices == batch.indices
    assert batch.with_view(layout=None, metadata=None) is batch


@pytest.mark.parametrize(
    "known_indices, message",
    [
        ("ab", "must be a sequence"),
        (5, "must be a sequence"),
        ([[0, 0]], "tuple"),
        ([(0, -1)], "non-negative"),
        ([(1, 0), (0, 0), (0, 2)], "does not extend parent"),
        ([(0,), (0, 0), (0, 2)], "at least 2"),
        ([(0, 0), (0, 2), (0, 2, 1)], "share one depth"),
        ([(0, 0), (0, 0), (0, 2)], "duplicated"),
        ([(0, 0), (0, 1)], "not among known_indices"),
        ([(0, 2), (0, 0)], "same order"),
    ],
)
def test_invalid_known_domain_is_rejected(known_indices, message) -> None:
    with pytest.raises(ContractError, match=message):
        Batch(
            [10, 20],
            indices=[(0, 0), (0, 2)],
            known_indices=known_indices,
            parent_index=(0,),
        )


def test_survivors_deeper_than_the_domain_are_rejected() -> None:
    with pytest.raises(ContractError, match="not among known_indices"):
        Batch([10], indices=[(0, 0, 0)], known_indices=[(0, 0)], parent_index=(0,))
