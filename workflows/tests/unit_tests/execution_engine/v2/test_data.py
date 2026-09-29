from fractions import Fraction
from types import MappingProxyType

import pytest
from roboflow_workflows.execution_engine.v2.data import (
    AXIS_KINDS,
    Axis,
    Batch,
    EntryLayout,
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
    TimeSpan,
    Timestamp,
    WorkflowsBuffer,
    validate_entry,
)
from roboflow_workflows.execution_engine.v2.errors import ContractError


def _timestamp(ticks: int, clock_id: str = "clock") -> Timestamp:
    return Timestamp(ticks=ticks, time_base=Fraction(1, 30), clock_id=clock_id)


def _temporal(ticks: int) -> TemporalContext:
    return TemporalContext(observed_coverage=_timestamp(ticks))


def _sample_layout() -> EntryLayout:
    return EntryLayout(axes=(Axis(id="samples", kind="sample", stationary=True),))


def _nested_layout() -> EntryLayout:
    return EntryLayout(
        axes=(
            Axis(id="samples", kind="sample", stationary=True),
            Axis(id="crop.regions", kind="dynamic_nesting"),
        )
    )


def _ragged_tree() -> Batch:
    # Child counts [2, 0, 1]; the middle parent is a genuine empty group.
    return Batch(
        [
            Batch(["a0", "a1"], indices=[(0, 0), (0, 1)], parent_index=(0,)),
            Batch.empty(parent_index=(1,)),
            Batch(["c0"], indices=[(2, 0)], parent_index=(2,)),
        ],
        indices=[(0,), (1,), (2,)],
    )


# --- Axis --------------------------------------------------------------------


def test_axis_kinds_are_the_four_agreed_kinds() -> None:
    assert AXIS_KINDS == ("sample", "static_nesting", "dynamic_nesting", "time")


def test_axis_rejects_unknown_kind_and_empty_id() -> None:
    with pytest.raises(ContractError, match="unknown kind"):
        Axis(id="x", kind="spatial")
    with pytest.raises(ContractError, match="non-empty string"):
        Axis(id="", kind="sample")


def test_static_nesting_implies_stationary() -> None:
    axis = Axis(id="regions", kind="static_nesting")

    assert axis.stationary is True


def test_dynamic_nesting_cannot_be_declared_stationary() -> None:
    with pytest.raises(ContractError, match="cannot be stationary"):
        Axis(id="regions", kind="dynamic_nesting", stationary=True)


def test_sample_stationarity_is_explicit() -> None:
    assert Axis(id="n", kind="sample").stationary is False
    assert Axis(id="n", kind="sample", stationary=True).stationary is True


# --- EntryLayout ---------------------------------------------------------------


def test_layout_defaults_to_ungrouped() -> None:
    layout = EntryLayout()

    assert layout.axes == ()
    assert layout.depth == 0
    assert layout.has_time is False


def test_layout_accepts_list_and_stores_tuple() -> None:
    layout = EntryLayout(axes=[Axis(id="n", kind="sample")])

    assert isinstance(layout.axes, tuple)
    assert layout.axis_ids == ("n",)


def test_layout_rejects_duplicate_axis_ids() -> None:
    with pytest.raises(ContractError, match="duplicate axis id 'c'"):
        EntryLayout(
            axes=(
                Axis(id="c", kind="dynamic_nesting"),
                Axis(id="c", kind="dynamic_nesting"),
            )
        )


def test_layout_rejects_sample_axis_not_first() -> None:
    with pytest.raises(ContractError, match="must be the first axis"):
        EntryLayout(
            axes=(Axis(id="c", kind="dynamic_nesting"), Axis(id="n", kind="sample"))
        )


def test_layout_rejects_two_time_axes() -> None:
    with pytest.raises(ContractError, match="at most one time axis"):
        EntryLayout(axes=(Axis(id="t1", kind="time"), Axis(id="t2", kind="time")))


def test_layout_requires_stationary_axes_before_time() -> None:
    with pytest.raises(
        ContractError, match="precedes time axis 't' and must be stationary"
    ):
        EntryLayout(
            axes=(
                Axis(id="n", kind="sample", stationary=True),
                Axis(id="c", kind="dynamic_nesting"),
                Axis(id="t", kind="time"),
            )
        )
    with pytest.raises(ContractError, match="'n' precedes time axis"):
        EntryLayout(axes=(Axis(id="n", kind="sample"), Axis(id="t", kind="time")))


def test_layout_can_represent_temporal_layouts() -> None:
    layout = EntryLayout(
        axes=(
            Axis(id="n", kind="sample", stationary=True),
            Axis(id="regions", kind="static_nesting"),
            Axis(id="t", kind="time"),
            Axis(id="c", kind="dynamic_nesting"),
        )
    )

    assert layout.has_time is True
    assert layout.depth == 4


def test_layout_rejects_non_axis_elements() -> None:
    with pytest.raises(ContractError, match="must be an Axis"):
        EntryLayout(axes=("n",))
    with pytest.raises(ContractError, match="sequence of Axis"):
        EntryLayout(axes=Axis(id="n", kind="sample"))


def test_layout_append_and_remove_last_axis_return_new_layouts() -> None:
    base = _sample_layout()

    extended = base.append_axis(Axis(id="c", kind="dynamic_nesting"))
    prefix = extended.remove_last_axis()

    assert extended.axis_ids == ("samples", "c")
    assert prefix == base
    assert base.axis_ids == ("samples",)
    assert extended.last_axis.id == "c"


def test_layout_remove_last_axis_on_ungrouped_fails() -> None:
    with pytest.raises(ContractError, match="ungrouped"):
        EntryLayout().remove_last_axis()
    with pytest.raises(ContractError, match="no last axis"):
        EntryLayout().last_axis


def test_layout_append_axis_revalidates_invariants() -> None:
    with pytest.raises(ContractError, match="duplicate axis id"):
        _sample_layout().append_axis(Axis(id="samples", kind="dynamic_nesting"))


# --- Temporal and sample context -------------------------------------------------


def test_timestamp_validation_and_seconds() -> None:
    timestamp = Timestamp(ticks=90, time_base=Fraction(1, 30), clock_id="c")

    assert timestamp.seconds == Fraction(3)
    assert Timestamp(ticks=1, time_base=2, clock_id="c").time_base == Fraction(2)
    with pytest.raises(ContractError, match="ticks must be an int"):
        Timestamp(ticks=1.5, time_base=Fraction(1), clock_id="c")
    with pytest.raises(ContractError, match="must be positive"):
        Timestamp(ticks=1, time_base=Fraction(0), clock_id="c")
    with pytest.raises(ContractError, match="clock_id"):
        Timestamp(ticks=1, time_base=Fraction(1), clock_id="")


def test_timespan_requires_same_clock_and_positive_length() -> None:
    TimeSpan(start=_timestamp(0), end=_timestamp(1))
    with pytest.raises(ContractError, match="share a clock"):
        TimeSpan(start=_timestamp(0, "a"), end=_timestamp(1, "b"))
    with pytest.raises(ContractError, match="end must be after start"):
        TimeSpan(start=_timestamp(1), end=_timestamp(1))


def test_temporal_context_requires_coverage_types() -> None:
    context = TemporalContext(
        observed_coverage=TimeSpan(start=_timestamp(0), end=_timestamp(2)),
        media_coverage=_timestamp(5),
    )

    assert context.capture_coverage is None
    with pytest.raises(ContractError, match="observed_coverage"):
        TemporalContext(observed_coverage=None)
    with pytest.raises(ContractError, match="media_coverage"):
        TemporalContext(observed_coverage=_timestamp(0), media_coverage=3)


def test_sample_context_snapshots_and_freezes_metadata() -> None:
    supplied = {"camera": {"lens": "wide"}, "tags": ["a", "b"], "ids": {1, 2}}

    context = SampleContext(source_id="cam-1", source_metadata=supplied)
    supplied["camera"]["lens"] = "tele"
    supplied["new"] = True

    assert context.source_type == "static"
    assert context.source_metadata["camera"]["lens"] == "wide"
    assert "new" not in context.source_metadata
    assert context.source_metadata["tags"] == ("a", "b")
    assert context.source_metadata["ids"] == frozenset({1, 2})
    with pytest.raises(TypeError):
        context.source_metadata["camera"] = {}
    with pytest.raises(TypeError):
        context.source_metadata["camera"]["lens"] = "x"


def test_sample_context_does_not_copy_leaf_payload_objects() -> None:
    class Payload:
        pass

    payload = Payload()

    context = SampleContext(source_id="s", source_metadata={"thumb": payload})

    assert context.source_metadata["thumb"] is payload


def test_sample_context_validation() -> None:
    with pytest.raises(ContractError, match="source_id"):
        SampleContext(source_id="")
    with pytest.raises(ContractError, match="source_type"):
        SampleContext(source_id="s", source_type="")
    with pytest.raises(ContractError, match="must be a mapping"):
        SampleContext(source_id="s", source_metadata=["x"])


def test_sample_context_equality_ignores_hash_but_compares_metadata() -> None:
    first = SampleContext(source_id="s", source_metadata={"a": 1})
    second = SampleContext(source_id="s", source_metadata={"a": 1})
    third = SampleContext(source_id="s", source_metadata={"a": 2})

    assert first == second
    assert first != third
    assert hash(first) == hash(second)


# --- EntryMetadata -------------------------------------------------------------


def test_metadata_default_is_empty() -> None:
    metadata = EntryMetadata()

    assert metadata.is_empty is True
    assert metadata.max_depth == 0
    assert metadata.sample_at((0, 1)) is None
    assert metadata.temporal_at(()) is None


def test_metadata_longest_prefix_lookup() -> None:
    root = SampleContext(source_id="root")
    child = SampleContext(source_id="child")
    metadata = EntryMetadata(sample={(): root, (0, 1): child})

    assert metadata.sample_at(()) is root
    assert metadata.sample_at((0,)) is root
    assert metadata.sample_at((0, 1)) is child
    assert metadata.sample_at((0, 1, 7)) is child
    assert metadata.sample_at((0, 2)) is root
    assert metadata.sample_at((3, 1)) is root


def test_metadata_explicit_none_stops_inheritance() -> None:
    root = SampleContext(source_id="root")
    metadata = EntryMetadata(sample={(): root, (1,): None})

    assert metadata.sample_at((0,)) is root
    assert metadata.sample_at((1,)) is None
    assert metadata.sample_at((1, 4)) is None
    assert metadata.max_depth == 1


def test_metadata_temporal_override_replaces_whole_context() -> None:
    root = TemporalContext(
        observed_coverage=_timestamp(0), media_coverage=_timestamp(10)
    )
    override = _temporal(5)
    metadata = EntryMetadata(temporal={(): root, (2,): override})

    resolved = metadata.temporal_at((2, 0))

    assert resolved is override
    assert resolved.media_coverage is None


def test_metadata_sample_and_temporal_maps_are_independent() -> None:
    metadata = EntryMetadata(
        sample={(): SampleContext(source_id="s"), (0,): None},
        temporal={(0,): _temporal(1)},
    )

    assert metadata.sample_at((0, 0)) is None
    assert metadata.temporal_at((0, 0)) == _temporal(1)
    assert metadata.temporal_at((1,)) is None
    assert metadata.sample_at((1,)).source_id == "s"


def test_metadata_snapshot_isolates_from_supplied_dict_and_is_read_only() -> None:
    supplied = {(): SampleContext(source_id="s")}

    metadata = EntryMetadata(sample=supplied)
    supplied[(0,)] = None
    supplied.clear()

    assert metadata.sample_at((0,)).source_id == "s"
    with pytest.raises(TypeError):
        metadata.sample[(0,)] = None
    with pytest.raises(AttributeError):
        metadata.sample = {}


def test_metadata_rejects_malformed_paths_and_values() -> None:
    with pytest.raises(ContractError, match="tuple of non-negative integers"):
        EntryMetadata(sample={"0": None})
    with pytest.raises(ContractError, match="tuple of non-negative integers"):
        EntryMetadata(sample={0: None})
    with pytest.raises(ContractError, match="only integers"):
        EntryMetadata(temporal={(0, "1"): None})
    with pytest.raises(ContractError, match="only integers"):
        EntryMetadata(temporal={(True,): None})
    with pytest.raises(ContractError, match="non-negative"):
        EntryMetadata(sample={(-1,): None})
    with pytest.raises(ContractError, match="must be a SampleContext or None"):
        EntryMetadata(sample={(): _temporal(0)})
    with pytest.raises(ContractError, match="must be a TemporalContext or None"):
        EntryMetadata(temporal={(): SampleContext(source_id="s")})
    with pytest.raises(ContractError, match="must be a mapping"):
        EntryMetadata(sample=[((), None)])


def test_metadata_lookup_rejects_malformed_index() -> None:
    with pytest.raises(ContractError, match="Lookup index"):
        EntryMetadata().sample_at([0])
    with pytest.raises(ContractError, match="Lookup index"):
        EntryMetadata().temporal_at((0, -1))


# --- Batch ---------------------------------------------------------------------


def test_batch_of_uses_local_one_component_indices() -> None:
    batch = Batch.of(["x", "y", "z"])

    assert batch.indices == ((0,), (1,), (2,))
    assert batch.content == ("x", "y", "z")
    assert batch.parent_index == ()
    assert batch.layout is None
    assert batch.metadata is None
    assert len(batch) == 3
    assert list(batch) == ["x", "y", "z"]
    assert batch[1] == "y"
    assert batch[-1] == "z"
    assert list(batch.iter_with_indices()) == [((0,), "x"), ((1,), "y"), ((2,), "z")]


def test_batch_membership_is_immutable_but_payloads_stay_mutable() -> None:
    payload = {"pixels": [1, 2]}
    source = [payload]

    batch = Batch.of(source)
    source.append({"pixels": [3]})
    payload["pixels"].append(9)

    assert len(batch) == 1
    assert batch[0] is payload
    assert batch[0]["pixels"] == [1, 2, 9]
    with pytest.raises(AttributeError):
        batch.content = ()
    with pytest.raises(TypeError):
        batch.content[0] = None
    with pytest.raises(TypeError):
        batch.indices[0] = (5,)


def test_batch_empty_is_a_valid_group_that_knows_its_parent() -> None:
    root_empty = Batch.empty()
    nested_empty = Batch.empty(parent_index=(1,))

    assert len(root_empty) == 0
    assert root_empty.indices == ()
    assert nested_empty.parent_index == (1,)
    assert nested_empty.indices == ()
    assert list(nested_empty.iter_with_indices()) == []


def test_batch_empty_differs_from_empty_list_and_none_payloads() -> None:
    assert isinstance(Batch.empty(), Batch)
    assert not isinstance([], Batch)
    assert Batch.empty() != []
    assert Batch.of([[]])[0] == []
    assert Batch.of([None])[0] is None
    assert len(Batch.of([[]])) == 1


def test_batch_rejects_non_group_content_types() -> None:
    with pytest.raises(ContractError, match="never implicit groups"):
        Batch("abc")
    with pytest.raises(ContractError, match="never implicit groups"):
        Batch({"a": 1})
    with pytest.raises(ContractError, match="never implicit groups"):
        Batch(None)
    with pytest.raises(ContractError, match="must be iterable"):
        Batch(42)


def test_batch_sparse_explicit_indices_are_retained_in_order() -> None:
    batch = Batch.of(["r2", "r0"], indices=[(2,), (0,)])

    assert batch.indices == ((2,), (0,))
    assert list(batch.iter_with_indices()) == [((2,), "r2"), ((0,), "r0")]


def test_batch_singleton_keeps_its_index() -> None:
    batch = Batch(["only"], indices=[(3, 7)], parent_index=(3,))

    assert len(batch) == 1
    assert batch.indices == ((3, 7),)


def test_batch_default_indices_extend_parent_index() -> None:
    batch = Batch(["a", "b"], parent_index=(4,))

    assert batch.indices == ((4, 0), (4, 1))


def test_batch_rejects_index_count_mismatch() -> None:
    with pytest.raises(ContractError, match="2 elements but 1 indices"):
        Batch.of(["a", "b"], indices=[(0,)])


def test_batch_rejects_malformed_indices() -> None:
    with pytest.raises(ContractError, match="tuple of non-negative integers"):
        Batch.of(["a"], indices=[0])
    with pytest.raises(ContractError, match="only integers"):
        Batch.of(["a"], indices=[("0",)])
    with pytest.raises(ContractError, match="non-negative"):
        Batch.of(["a"], indices=[(-1,)])
    with pytest.raises(ContractError, match="sequence of tuples"):
        Batch.of(["a"], indices=5)
    with pytest.raises(ContractError, match="parent_index"):
        Batch(["a"], parent_index=[0])


def test_batch_rejects_indices_not_extending_parent() -> None:
    with pytest.raises(ContractError, match="must have 2 component"):
        Batch(["a"], indices=[(0,)], parent_index=(1,))
    with pytest.raises(ContractError, match="does not extend parent index"):
        Batch(["a"], indices=[(0, 0)], parent_index=(1,))
    with pytest.raises(ContractError, match="must have 1 component"):
        Batch.of(["a"], indices=[(0, 0)])


def test_batch_rejects_duplicate_indices() -> None:
    with pytest.raises(ContractError, match="duplicated"):
        Batch.of(["a", "b"], indices=[(1,), (1,)])


def test_batch_positional_indexing_only_accepts_ints() -> None:
    batch = Batch.of(["a"])

    with pytest.raises(TypeError):
        batch[(0,)]
    with pytest.raises(TypeError):
        batch[True]
    with pytest.raises(IndexError):
        batch[1]


def test_batch_equality_compares_membership_not_views() -> None:
    plain = Batch.of(["a", "b"])
    viewed = plain.with_view(layout=_sample_layout(), metadata=EntryMetadata())

    assert plain == Batch.of(["a", "b"])
    assert plain == viewed
    assert plain != Batch.of(["a", "b"], indices=[(1,), (0,)])
    assert plain != Batch(["a", "b"], parent_index=(0,))
    assert plain != ["a", "b"]
    with pytest.raises(TypeError):
        hash(plain)


def test_batch_with_view_shares_membership_and_reuses_matching_view() -> None:
    layout = _sample_layout()
    metadata = EntryMetadata(sample={(): SampleContext(source_id="s")})
    plain = Batch.of(["a"])

    viewed = plain.with_view(layout=layout, metadata=metadata)

    assert viewed is not plain
    assert viewed.content is plain.content
    assert viewed.indices is plain.indices
    assert viewed.layout is layout
    assert viewed.metadata is metadata
    assert viewed.with_view(layout=layout, metadata=metadata) is viewed
    assert plain.layout is None


def test_batch_rejects_wrong_view_types() -> None:
    with pytest.raises(ContractError, match="layout must be an EntryLayout"):
        Batch.of(["a"]).with_view(layout="[N]", metadata=None)
    with pytest.raises(ContractError, match="metadata must be an EntryMetadata"):
        Batch(["a"], metadata={})


def test_batch_generator_content_is_consumed_once() -> None:
    batch = Batch(value for value in range(3))

    assert batch.content == (0, 1, 2)
    assert list(batch) == [0, 1, 2]


def test_batch_repr_mentions_indices_and_parent() -> None:
    text = repr(Batch(["a"], parent_index=(2,)))

    assert "parent_index=(2,)" in text
    assert "indices=((2, 0),)" in text


# --- InputValue ----------------------------------------------------------------


def test_input_value_defaults_to_empty_metadata() -> None:
    value = InputValue(data=[1, 2])

    assert value.data == [1, 2]
    assert value.metadata.is_empty is True
    assert InputValue(data=None).data is None


def test_input_value_rejects_non_metadata() -> None:
    with pytest.raises(ContractError, match="EntryMetadata"):
        InputValue(data=1, metadata={(): None})


def test_input_value_is_frozen() -> None:
    value = InputValue(data=1)

    with pytest.raises(AttributeError):
        value.data = 2


# --- validate_entry ------------------------------------------------------------


def test_validate_entry_accepts_ungrouped_payloads_including_none_and_lists() -> None:
    for payload in (None, [], [1, 2], {"a": 1}, "text", 0):
        validate_entry(payload, layout=EntryLayout(), metadata=EntryMetadata())


def test_validate_entry_rejects_batch_for_ungrouped_layout() -> None:
    with pytest.raises(ContractError, match="declares no further grouping axis"):
        validate_entry(Batch.of([1]), layout=EntryLayout(), metadata=EntryMetadata())


def test_validate_entry_rejects_plain_list_for_grouped_layout() -> None:
    with pytest.raises(ContractError, match="plain lists are payloads, not groups"):
        validate_entry([1, 2], layout=_sample_layout(), metadata=EntryMetadata())


def test_validate_entry_accepts_ragged_tree_with_empty_group() -> None:
    validate_entry(_ragged_tree(), layout=_nested_layout(), metadata=EntryMetadata())


def test_validate_entry_accepts_singleton_and_sparse_indices() -> None:
    sparse = Batch(
        [Batch(["x"], indices=[(5, 3)], parent_index=(5,))],
        indices=[(5,)],
    )

    validate_entry(sparse, layout=_nested_layout(), metadata=EntryMetadata())


def test_validate_entry_accepts_empty_root_batch_at_any_depth() -> None:
    validate_entry(Batch.empty(), layout=_sample_layout(), metadata=EntryMetadata())
    validate_entry(Batch.empty(), layout=_nested_layout(), metadata=EntryMetadata())


def test_validate_entry_rejects_too_shallow_tree() -> None:
    with pytest.raises(
        ContractError, match=r"index \(0,\) must be a Batch \(remaining depth 1\)"
    ):
        validate_entry(
            Batch.of(["leaf"]), layout=_nested_layout(), metadata=EntryMetadata()
        )


def test_validate_entry_rejects_too_deep_tree() -> None:
    with pytest.raises(ContractError, match=r"index \(0,\) is a Batch"):
        validate_entry(
            Batch([Batch(["x"], parent_index=(0,))]),
            layout=_sample_layout(),
            metadata=EntryMetadata(),
        )


def test_validate_entry_rejects_nested_batch_with_local_indices() -> None:
    # Block-style local indices inside an EE tree do not form full paths.
    tree = Batch([Batch.of(["x"])])

    with pytest.raises(ContractError, match=r"reports parent_index \(\)"):
        validate_entry(tree, layout=_nested_layout(), metadata=EntryMetadata())


def test_validate_entry_rejects_root_batch_with_parent_index() -> None:
    with pytest.raises(ContractError, match=r"index \(\) reports parent_index \(1,\)"):
        validate_entry(
            Batch(["x"], parent_index=(1,)),
            layout=_sample_layout(),
            metadata=EntryMetadata(),
        )


def test_validate_entry_three_levels_deep() -> None:
    layout = EntryLayout(
        axes=(
            Axis(id="n", kind="sample"),
            Axis(id="c", kind="dynamic_nesting"),
            Axis(id="d", kind="dynamic_nesting"),
        )
    )
    tree = Batch(
        [
            Batch(
                [
                    Batch(["x", "y"], parent_index=(0, 0)),
                    Batch.empty(parent_index=(0, 1)),
                ],
                parent_index=(0,),
            ),
            Batch.empty(parent_index=(1,)),
        ]
    )
    metadata = EntryMetadata(
        sample={
            (): SampleContext(source_id="s"),
            (0, 0, 1): None,
            (0, 1): None,
            (1,): None,
        }
    )

    validate_entry(tree, layout=layout, metadata=metadata)


def test_validate_entry_metadata_paths_must_address_existing_positions() -> None:
    tree = _ragged_tree()
    layout = _nested_layout()

    validate_entry(
        tree,
        layout=layout,
        metadata=EntryMetadata(sample={(): None, (1,): None, (2, 0): None}),
    )
    with pytest.raises(
        ContractError, match=r"'sample' index \(1, 0\) does not address"
    ):
        validate_entry(
            tree, layout=layout, metadata=EntryMetadata(sample={(1, 0): None})
        )
    with pytest.raises(
        ContractError, match=r"'temporal' index \(3,\) does not address"
    ):
        validate_entry(
            tree, layout=layout, metadata=EntryMetadata(temporal={(3,): None})
        )


def test_validate_entry_metadata_paths_must_not_exceed_layout_depth() -> None:
    with pytest.raises(ContractError, match="deeper than the layout depth 0"):
        validate_entry(
            "payload",
            layout=EntryLayout(),
            metadata=EntryMetadata(sample={(0,): None}),
        )
    with pytest.raises(ContractError, match="deeper than the layout depth 1"):
        validate_entry(
            Batch.of(["a"]),
            layout=_sample_layout(),
            metadata=EntryMetadata(temporal={(0, 0): _temporal(0)}),
        )


def test_validate_entry_root_metadata_is_always_allowed() -> None:
    validate_entry(
        "payload",
        layout=EntryLayout(),
        metadata=EntryMetadata(
            sample={(): SampleContext(source_id="s")}, temporal={(): None}
        ),
    )


def test_validate_entry_requires_layout_and_metadata_types() -> None:
    with pytest.raises(ContractError, match="layout must be an EntryLayout"):
        validate_entry(1, layout=(), metadata=EntryMetadata())
    with pytest.raises(ContractError, match="metadata must be an EntryMetadata"):
        validate_entry(1, layout=EntryLayout(), metadata={})


# --- WorkflowsBuffer -----------------------------------------------------------


def test_buffer_holds_entries_with_independent_layouts() -> None:
    tree = _ragged_tree()
    summary = Batch.of(["s0", "s1", "s2"])
    crops_metadata = EntryMetadata(sample={(): SampleContext(source_id="s")})

    buffer = WorkflowsBuffer(
        lineage_id="crop",
        pulse_id=1,
        data={"crops": tree, "summary": summary},
        layout={"crops": _nested_layout(), "summary": _sample_layout()},
        metadata={"crops": crops_metadata, "summary": EntryMetadata()},
    )

    assert buffer.entry_names == ("crops", "summary")
    assert buffer.is_filtered is False
    assert buffer.layout["crops"].depth == 2
    assert buffer.layout["summary"].depth == 1
    assert buffer.data["crops"] is tree
    assert buffer.metadata["crops"] is crops_metadata


def test_buffer_snapshots_supplied_dicts_and_is_read_only() -> None:
    data = {"x": 1}
    layout = {"x": EntryLayout()}
    metadata = {"x": EntryMetadata()}

    buffer = WorkflowsBuffer(
        lineage_id="l", pulse_id=0, data=data, layout=layout, metadata=metadata
    )
    data["y"] = 2
    layout["y"] = EntryLayout()

    assert buffer.entry_names == ("x",)
    with pytest.raises(TypeError):
        buffer.data["z"] = 3
    with pytest.raises(AttributeError):
        buffer.data = {}


def test_buffer_rejects_mismatched_keys() -> None:
    with pytest.raises(ContractError, match="equal keys"):
        WorkflowsBuffer(
            lineage_id="l",
            pulse_id=0,
            data={"x": 1},
            layout={"x": EntryLayout()},
            metadata={},
        )
    with pytest.raises(ContractError, match="equal keys"):
        WorkflowsBuffer(
            lineage_id="l",
            pulse_id=0,
            data={"x": 1},
            layout={"y": EntryLayout()},
            metadata={"x": EntryMetadata()},
        )


def test_buffer_validates_each_entry_and_names_the_entry() -> None:
    with pytest.raises(
        ContractError, match="Entry 'crops': .*plain lists are payloads"
    ):
        WorkflowsBuffer(
            lineage_id="l",
            pulse_id=0,
            data={"crops": [1]},
            layout={"crops": _sample_layout()},
            metadata={"crops": EntryMetadata()},
        )
    with pytest.raises(ContractError, match="Entry 'x': layout must be an EntryLayout"):
        WorkflowsBuffer(
            lineage_id="l",
            pulse_id=0,
            data={"x": 1},
            layout={"x": None},
            metadata={"x": EntryMetadata()},
        )
    with pytest.raises(
        ContractError, match="Entry 'x': metadata must be an EntryMetadata"
    ):
        WorkflowsBuffer(
            lineage_id="l",
            pulse_id=0,
            data={"x": 1},
            layout={"x": EntryLayout()},
            metadata={"x": None},
        )


def test_buffer_entry_validation_preserves_cause() -> None:
    with pytest.raises(ContractError) as info:
        WorkflowsBuffer(
            lineage_id="l",
            pulse_id=0,
            data={"x": Batch.of([1])},
            layout={"x": EntryLayout()},
            metadata={"x": EntryMetadata()},
        )

    assert isinstance(info.value.__cause__, ContractError)


def test_filtered_buffer_keeps_identity_with_all_dicts_empty() -> None:
    buffer = WorkflowsBuffer.filtered(lineage_id="gate", pulse_id=7)

    assert buffer.is_filtered is True
    assert buffer.lineage_id == "gate"
    assert buffer.pulse_id == 7
    assert dict(buffer.data) == {}
    assert dict(buffer.layout) == {}
    assert dict(buffer.metadata) == {}


def test_empty_payload_entry_is_not_a_filtered_buffer() -> None:
    buffer = WorkflowsBuffer(
        lineage_id="l",
        pulse_id=0,
        data={"detections": [], "group": Batch.empty(), "value": None},
        layout={
            "detections": EntryLayout(),
            "group": _sample_layout(),
            "value": EntryLayout(),
        },
        metadata={
            "detections": EntryMetadata(),
            "group": EntryMetadata(),
            "value": EntryMetadata(),
        },
    )

    assert buffer.is_filtered is False
    assert buffer.data["detections"] == []
    assert len(buffer.data["group"]) == 0
    assert buffer.data["value"] is None


def test_buffer_rejects_bad_identity_and_names() -> None:
    with pytest.raises(ContractError, match="lineage_id"):
        WorkflowsBuffer(lineage_id="", pulse_id=0)
    with pytest.raises(ContractError, match="pulse_id must be an int"):
        WorkflowsBuffer(lineage_id="l", pulse_id="1")
    with pytest.raises(ContractError, match="pulse_id must be an int"):
        WorkflowsBuffer(lineage_id="l", pulse_id=True)
    with pytest.raises(ContractError, match="entry name"):
        WorkflowsBuffer(
            lineage_id="l",
            pulse_id=0,
            data={"": 1},
            layout={"": EntryLayout()},
            metadata={"": EntryMetadata()},
        )
    with pytest.raises(ContractError, match="data must be a mapping"):
        WorkflowsBuffer(lineage_id="l", pulse_id=0, data=[1], layout={}, metadata={})


def test_buffer_metadata_paths_checked_against_ragged_tree() -> None:
    with pytest.raises(
        ContractError, match=r"Entry 'crops': Metadata 'sample' index \(1, 0\)"
    ):
        WorkflowsBuffer(
            lineage_id="l",
            pulse_id=0,
            data={"crops": _ragged_tree()},
            layout={"crops": _nested_layout()},
            metadata={"crops": EntryMetadata(sample={(1, 0): None})},
        )


def test_context_attached_to_batch_cannot_be_mutated_through_view() -> None:
    supplied_sample = {(): SampleContext(source_id="s")}
    metadata = EntryMetadata(sample=supplied_sample)
    batch = Batch.of(["a"]).with_view(layout=_sample_layout(), metadata=metadata)
    supplied_sample[()] = None

    assert batch.metadata.sample_at((0,)).source_id == "s"
    with pytest.raises(TypeError):
        batch.metadata.sample[()] = None
    with pytest.raises(AttributeError):
        batch.layout.axes = ()


def test_source_metadata_snapshots_nested_proxy_backing_and_children() -> None:
    backing = {"regions": [1, 2]}
    sample = SampleContext(
        "camera",
        source_metadata={"nested": MappingProxyType(backing)},
    )

    backing["regions"].append(3)
    backing["other"] = "later"

    nested = sample.source_metadata["nested"]
    assert dict(nested) == {"regions": (1, 2)}
    with pytest.raises(TypeError):
        nested["other"] = "mutation"
