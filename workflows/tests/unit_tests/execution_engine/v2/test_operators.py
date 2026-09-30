"""Built-in operator algorithms driven directly, without a runtime.

``Align`` and ``Window`` receive arrivals as the active runtime would hand
them over: one ``push`` per upstream pulse, ``end_input`` when a domain
ends, ``finish`` once, then ``close``.
"""

import itertools
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

import pytest
from roboflow_workflows.execution_engine.v2.data import (
    AXIS_KIND_DYNAMIC_NESTING,
    AXIS_KIND_SAMPLE,
    AXIS_KIND_STATIC_NESTING,
    AXIS_KIND_TIME,
    Axis,
    EntryLayout,
    EntryMetadata,
    SampleContext,
    TemporalContext,
    TimeSpan,
    Timestamp,
)
from roboflow_workflows.execution_engine.v2.errors import (
    OperatorError,
    OperatorInputError,
    ParamsValidationError,
    WorkflowCompileError,
)
from roboflow_workflows.execution_engine.v2.execution.entries import Entry, scalar_entry
from roboflow_workflows.execution_engine.v2.kinds import Kind
from roboflow_workflows.execution_engine.v2.operators import (
    Arrival,
    Operator,
    OperatorDeclarationError,
    OperatorInput,
    OperatorPulse,
    spec_of_operator,
)
from roboflow_workflows.execution_engine.v2.operators.alignment import Align
from roboflow_workflows.execution_engine.v2.operators.window import Window
from roboflow_workflows.execution_engine.v2.plan import PulseKey

CLOCK = "media"
FLOAT = Kind(name="float", description="A float.")
IMAGE = Kind(name="image", description="An image.")
OBSERVED = Timestamp(ticks=0, time_base=Fraction(1, 1000), clock_id="engine")


def at(ms: Any, *, clock: str = CLOCK) -> Timestamp:
    """A media timestamp in milliseconds; a Fraction for exact rational ticks."""
    if isinstance(ms, Fraction):
        return Timestamp(
            ticks=ms.numerator,
            time_base=Fraction(1, 1000 * ms.denominator),
            clock_id=clock,
        )

    return Timestamp(ticks=ms, time_base=Fraction(1, 1000), clock_id=clock)


def sample(
    payload: Any, ms: Any = None, *, source: str = "cam", coverage: Any = None
) -> Entry:
    media = coverage if coverage is not None else (at(ms) if ms is not None else None)
    metadata = EntryMetadata(
        sample={(): SampleContext(source_id=source)},
        temporal={
            (): TemporalContext(observed_coverage=OBSERVED, media_coverage=media)
        },
    )

    return scalar_entry(payload, metadata=metadata)


def absent(layout: EntryLayout = EntryLayout()) -> Entry:
    return Entry(
        layout=layout,
        metadata=EntryMetadata(),
        children={},
        values={},
        filtered=frozenset({()}),
    )


def key(domain: str, sequence: int) -> PulseKey:
    return PulseKey(active_run_id="run", source=domain, sequence=sequence)


def arrive(
    name: str, entry: Entry, sequence: int, *, domain: Optional[str] = None
) -> Arrival:
    return Arrival(input=name, entry=entry, pulse=key(domain or name, sequence))


def payloads(pulses: Sequence[OperatorPulse]) -> List[Dict[str, Any]]:
    return [
        {name: entry.values[()] for name, entry in pulse.ports.items()}
        for pulse in pulses
    ]


def aligner(names: Sequence[str] = ("a", "b"), **params: Any) -> Align:
    spec = spec_of_operator(Align)
    validated = spec.validate_params({"clock": CLOCK, **params}, operator_name="pair")
    inputs = [OperatorInput(name, "input", EntryLayout(), (FLOAT,)) for name in names]
    spec.plan_ports("pair", validated, inputs)

    return Align(name="pair", params=validated, inputs=inputs)


# Contract ------------------------------------------------------------------


def test_builtin_specs_are_class_owned() -> None:
    align = spec_of_operator(Align)
    window = spec_of_operator(Window)

    assert (align.type, align.input_roles) == ("v2/align@v1", ("input",))
    assert (window.type, window.input_roles) == ("v2/window@v1", ("collect", "hold"))
    assert align.describe()["params_schema"]["required"] == ["clock"]
    assert align.identities == ("v2/align@v1",)


def test_params_are_literals_and_ignore_declaration_keys() -> None:
    spec = spec_of_operator(Window)

    params = spec.validate_params(
        {
            "type": "v2/window@v1",
            "name": "w",
            "collect": {"x": "$sources.a.x"},
            "size": 2,
        }
    )

    assert (params.size, params.step, params.partial) == (2, None, "drop")
    with pytest.raises(ParamsValidationError, match="literals"):
        spec.validate_params({"size": "$inputs.size"}, operator_name="w")
    with pytest.raises(ParamsValidationError, match=r"step 3 is larger than size 2"):
        spec.validate_params({"size": 2, "step": 3}, operator_name="w")
    with pytest.raises(ParamsValidationError, match="size"):
        spec.validate_params({"size": "3"}, operator_name="w")


def test_abstract_and_incomplete_operator_classes_are_rejected() -> None:
    class Abstract(Operator):
        pass

    with pytest.raises(OperatorDeclarationError, match="abstract"):
        spec_of_operator(Abstract)
    with pytest.raises(OperatorDeclarationError, match="plan_ports"):

        class NoPorts(Operator):
            type = "test/no_ports@v1"
            input_roles = ("input",)

            def push(self, arrivals):
                return []

    with pytest.raises(OperatorDeclarationError, match="input_roles"):

        class NoRoles(Operator):
            type = "test/no_roles@v1"

            @classmethod
            def plan_ports(cls, name, params, inputs):
                return {}

            def push(self, arrivals):
                return []


def test_plan_ports_rejects_roles_the_operator_does_not_accept() -> None:
    spec = spec_of_operator(Align)
    params = spec.validate_params({"clock": CLOCK})

    with pytest.raises(
        WorkflowCompileError, match="does not accept collect input 'a'"
    ) as raised:
        spec.plan_ports("pair", params, [("a", "collect", EntryLayout(), ())])

    assert raised.value.step_path == ("$operators", "pair")


# Alignment -------------------------------------------------------------------


def test_align_plans_fields_or_one_stationary_batch() -> None:
    spec = spec_of_operator(Align)
    inputs = [
        ("a", "input", EntryLayout(), (IMAGE,)),
        ("b", "input", EntryLayout(), (IMAGE,)),
    ]

    fields = spec.plan_ports("pair", spec.validate_params({"clock": CLOCK}), inputs)
    batch = spec.plan_ports(
        "pair", spec.validate_params({"clock": CLOCK, "layout": "batch"}), inputs
    )

    assert {name: port.kind_names for name, port in fields.items()} == {
        "a": ("image",),
        "b": ("image",),
    }
    assert list(batch) == ["samples"]
    assert batch["samples"].layout.axes == (
        Axis("operators.pair:samples", AXIS_KIND_SAMPLE, stationary=True),
    )


@pytest.mark.parametrize(
    ("params", "inputs", "message"),
    [
        ({}, [("a", EntryLayout(), (FLOAT,))], "at least two inputs"),
        (
            {},
            [
                ("a", EntryLayout((Axis("n", AXIS_KIND_SAMPLE),)), (FLOAT,)),
                ("b", EntryLayout(), (FLOAT,)),
            ],
            "input 'a' has axes",
        ),
        (
            {"leader": "c"},
            [("a", EntryLayout(), (FLOAT,)), ("b", EntryLayout(), (FLOAT,))],
            "leader 'c'",
        ),
        (
            {"layout": "batch"},
            [("a", EntryLayout(), (FLOAT,)), ("b", EntryLayout(), (IMAGE,))],
            "same kinds",
        ),
    ],
)
def test_align_rejects_inputs_it_cannot_pair(params, inputs, message) -> None:
    spec = spec_of_operator(Align)
    validated = spec.validate_params({"clock": CLOCK, **params})
    described = [(name, "input", layout, kinds) for name, layout, kinds in inputs]

    with pytest.raises(WorkflowCompileError, match=message) as raised:
        spec.plan_ports("pair", validated, described)

    if isinstance(raised.value, OperatorInputError):
        assert raised.value.field_path == ("input", "a")


def test_leader_waits_until_every_follower_reaches_its_tolerance() -> None:
    operator = aligner(tolerance_ms=10)

    assert operator.push([arrive("a", sample("a0", 100), 0)]) == []
    assert operator.push([arrive("b", sample("b0", 95), 0)]) == []
    # Equality at the upper bound is enough: strict progression rules out 109.
    emitted = operator.push([arrive("b", sample("b1", 110), 1)])

    assert payloads(emitted) == [{"a": "a0", "b": "b0"}]
    assert emitted[0].causes == (key("a", 0), key("b", 0))


def test_nearest_within_inclusive_tolerance_ties_to_the_earlier_value() -> None:
    operator = aligner(tolerance_ms=10)

    operator.push([arrive("b", sample("b-early", 90), 0)])
    operator.push([arrive("b", sample("b-late", 110), 1)])
    emitted = operator.push([arrive("a", sample("a0", 100), 0)])

    assert payloads(emitted) == [{"a": "a0", "b": "b-early"}]


def test_exact_rational_timestamps_decide_ties() -> None:
    # Values 1/3 ms before and after the leader are exactly as near; float
    # seconds would round the two distances differently.
    operator = aligner(tolerance_ms=1)
    third = Fraction(1, 3)

    operator.push([arrive("b", sample("before", 100 - third), 0)])
    operator.push([arrive("b", sample("after", 100 + third), 1)])
    operator.push([arrive("a", sample("a0", 100), 0)])
    emitted = operator.push([arrive("b", sample("b2", 101), 2)])

    assert payloads(emitted) == [{"a": "a0", "b": "before"}]


def _stream_results(order: Sequence[str]) -> List[Dict[str, Any]]:
    leaders = iter([("a0", 0), ("a1", 40), ("a2", 80), ("a3", 120)])
    followers = iter([("b0", 5), ("b1", 33), ("b2", 66), ("b3", 99), ("b4", 125)])
    sequences = {"a": itertools.count(), "b": itertools.count()}
    operator = aligner(tolerance_ms=10)
    emitted: List[OperatorPulse] = []
    for name in order:
        payload, ms = next(leaders if name == "a" else followers)
        emitted += operator.push(
            [arrive(name, sample(payload, ms), next(sequences[name]))]
        )
    for name in ("b", "a"):
        emitted += operator.end_input(name)
    emitted += operator.finish("eof")

    return payloads(emitted)


def test_every_legal_reader_interleaving_produces_the_same_pairs() -> None:
    outcomes = set()
    for positions in itertools.combinations(range(9), 4):
        order = ["a" if index in positions else "b" for index in range(9)]
        outcomes.add(
            tuple(tuple(sorted(item.items())) for item in _stream_results(order))
        )

    assert outcomes == {
        (
            (("a", "a0"), ("b", "b0")),
            (("a", "a1"), ("b", "b1")),
            (("a", "a3"), ("b", "b4")),
        )
    }


def test_inputs_of_one_upstream_pulse_are_ingested_before_deciding() -> None:
    operator = aligner(("a", "b"), tolerance_ms=0)

    emitted = operator.push(
        [
            arrive("a", sample("a0", 10), 0, domain="both"),
            arrive("b", sample("b0", 10), 0, domain="both"),
        ]
    )

    assert payloads(emitted) == [{"a": "a0", "b": "b0"}]
    assert emitted[0].causes == (key("both", 0),)


def test_late_values_are_counted_and_discarded() -> None:
    operator = aligner(tolerance_ms=0)

    operator.push([arrive("b", sample("b0", 50), 0)])
    operator.push([arrive("b", sample("b-repeat", 50), 1)])
    operator.push([arrive("b", sample("b-back", 40), 2)])
    emitted = operator.push([arrive("a", sample("a0", 50), 0)])

    assert payloads(emitted) == [{"a": "a0", "b": "b0"}]
    assert operator.counters.late == 2


def test_eof_is_an_infinite_watermark_and_missing_policy_decides() -> None:
    dropping = aligner(tolerance_ms=5)
    partial = aligner(tolerance_ms=5, missing="partial")
    for operator in (dropping, partial):
        operator.push([arrive("a", sample("a0", 100), 0)])
        assert operator.push([arrive("b", sample("b0", 50), 0)]) == []

    assert dropping.end_input("b") == []
    assert dropping.counters.dropped == 1
    assert payloads(partial.end_input("b")) == [{"a": "a0"}]


def test_filtered_arrivals_carry_no_watermark_until_eof() -> None:
    operator = aligner(tolerance_ms=0, missing="partial")

    operator.push([arrive("a", sample("a0", 10), 0)])
    assert operator.push([arrive("b", absent(), 0)]) == []
    emitted = operator.end_input("b")

    assert payloads(emitted) == [{"a": "a0"}]
    assert operator.counters.filtered == 1
    # Arrivals are counted by the active runtime, not by a directly driven operator.
    assert operator.counters.arrivals == 0


def test_followers_ahead_of_the_leader_are_retained_not_cleared() -> None:
    operator = aligner(tolerance_ms=0)

    for index, ms in enumerate((10, 20, 30)):
        operator.push([arrive("b", sample(f"b{index}", ms), index)])
    emitted = operator.push([arrive("a", sample("a0", 30), 0)])

    assert payloads(emitted) == [{"a": "a0", "b": "b2"}]
    assert operator.counters.peak_retained == 3
    # Later leaders are after 30 ms, so with no tolerance every value is spent.
    assert operator.counters.evicted == 3


@pytest.mark.parametrize("follower", ["b", "c"])
def test_every_follower_is_bounded_and_overflow_fails_explicitly(follower: str) -> None:
    operator = aligner(("a", "b", "c"), max_pending=2)

    operator.push([arrive(follower, sample("x", 10), 0)])
    operator.push([arrive(follower, sample("y", 20), 1)])
    with pytest.raises(OperatorError, match="above max_pending=2") as raised:
        operator.push([arrive(follower, sample("z", 30), 2)])

    assert (raised.value.operator, raised.value.input) == ("pair", follower)
    assert "ahead of leader 'a'" in str(raised.value)


def test_leader_overflow_names_the_followers_it_waits_for() -> None:
    operator = aligner(("a", "b", "c"), max_pending=1)
    operator.push([arrive("c", sample("c0", 100), 0)])
    operator.push([arrive("a", sample("a0", 10), 0)])

    with pytest.raises(OperatorError, match=r"waits for \['b'\]") as raised:
        operator.push([arrive("a", sample("a1", 20), 1)])

    assert raised.value.input == "a"


def test_consume_once_pairs_each_follower_value_at_most_once() -> None:
    results = {}
    for consume in ("reuse", "once"):
        operator = aligner(tolerance_ms=20, consume=consume, missing="partial")
        operator.push([arrive("b", sample("b0", 10), 0)])
        operator.push([arrive("a", sample("a0", 5), 0)])
        operator.push([arrive("a", sample("a1", 15), 1)])
        results[consume] = payloads(operator.push([arrive("b", sample("b1", 100), 1)]))

    assert results["reuse"] == [{"a": "a0", "b": "b0"}, {"a": "a1", "b": "b0"}]
    assert results["once"] == [{"a": "a0", "b": "b0"}, {"a": "a1"}]


def test_stop_decides_waiting_leaders_with_the_missing_policy() -> None:
    operator = aligner(tolerance_ms=0, missing="partial")
    operator.push([arrive("a", sample("a0", 10), 0)])

    emitted = operator.finish("stop")

    assert payloads(emitted) == [{"a": "a0"}]


@pytest.mark.parametrize(
    ("coverage", "message"),
    [
        (None, "has no media_coverage"),
        (TimeSpan(at(0), at(10)), "TimeSpan media_coverage"),
        (at(10, clock="other"), "clock 'other'"),
    ],
)
def test_values_without_a_usable_timestamp_fail_with_attribution(
    coverage, message
) -> None:
    operator = aligner()
    entry = sample("b0", coverage=coverage) if coverage is not None else sample("b0")

    with pytest.raises(OperatorError, match=message) as raised:
        operator.push([arrive("b", entry, 3)])

    assert (raised.value.operator, raised.value.input) == ("pair", "b")
    assert "pulse b#3" in str(raised.value)


def test_fields_layout_keeps_entries_and_payload_identity() -> None:
    operator = aligner()
    frame = object()
    entry = sample(frame, 10, source="camera")

    operator.push([arrive("a", entry, 0)])
    emitted = operator.push([arrive("b", sample(1.0, 10, source="probe"), 0)])

    assert emitted[0].ports["a"] is entry
    assert emitted[0].ports["b"].metadata.sample_at(()).source_id == "probe"


def test_batch_layout_orders_members_and_keeps_missing_positions_filtered() -> None:
    operator = aligner(("a", "b", "c"), leader="b", layout="batch", missing="partial")
    frame = object()

    operator.push([arrive("b", sample(frame, 10, source="cam-b"), 0)])
    operator.push([arrive("a", sample("a0", 10, source="cam-a"), 0)])
    emitted = operator.end_input("c")

    batch = emitted[0].ports["samples"]
    assert batch.layout.axis_ids == ("operators.pair:samples",)
    assert batch.children[()] == ((0,), (1,), (2,))
    assert batch.values == {(0,): "a0", (1,): frame}
    assert batch.values[(1,)] is frame
    assert batch.filtered == frozenset({(2,)})
    assert batch.metadata.sample_at(()) is None
    assert batch.metadata.temporal_at(()) is None
    assert batch.metadata.sample_at((0,)).source_id == "cam-a"
    assert batch.metadata.temporal_at((1,)).media_coverage == at(10)
    assert batch.metadata.temporal_at((2,)) is None


def test_close_and_finish_release_retained_values() -> None:
    operator = aligner()
    operator.push([arrive("b", sample("b0", 10), 0)])
    operator.push([arrive("a", sample("a0", 50), 0)])

    operator.close()

    assert operator.end_input("b") == []


# Window ----------------------------------------------------------------------

N = Axis("sources.cam:n", AXIS_KIND_SAMPLE, stationary=True)
C = Axis("crop:regions", AXIS_KIND_STATIC_NESTING)
T = Axis("operators.clip:t", AXIS_KIND_TIME)


def windowed(
    collect: Dict[str, EntryLayout],
    hold: Optional[Dict[str, EntryLayout]] = None,
    **params: Any,
) -> Window:
    spec = spec_of_operator(Window)
    validated = spec.validate_params(params, operator_name="clip")
    inputs = [
        OperatorInput(name, "collect", layout, (IMAGE,))
        for name, layout in collect.items()
    ]
    inputs += [
        OperatorInput(name, "hold", layout, (IMAGE,))
        for name, layout in (hold or {}).items()
    ]
    spec.plan_ports("clip", validated, inputs)

    return Window(name="clip", params=validated, inputs=inputs)


def row(sequence: int, **entries: Entry) -> List[Arrival]:
    return [
        Arrival(input=name, entry=entry, pulse=key("cam", sequence))
        for name, entry in entries.items()
    ]


def grouped(
    values: Dict[tuple, Any], *, layout: EntryLayout, children, filtered=(), sample=None
) -> Entry:
    return Entry(
        layout=layout,
        metadata=EntryMetadata(
            sample=sample or {(): SampleContext("cam")},
            temporal={(): TemporalContext(observed_coverage=OBSERVED)},
        ),
        children=children,
        values=values,
        filtered=frozenset(filtered),
    )


def test_window_plans_a_trailing_time_axis_and_prefix_holds() -> None:
    spec = spec_of_operator(Window)
    params = spec.validate_params({"size": 3})
    ports = spec.plan_ports(
        "clip",
        params,
        [
            ("crops", "collect", EntryLayout((N, C)), (IMAGE,)),
            ("frame", "hold", EntryLayout((N,)), (IMAGE,)),
        ],
    )

    assert ports["crops"].layout.axes == (N, C, T)
    assert ports["frame"].layout.axes == (N,)


def test_window_needs_a_collect_input() -> None:
    spec = spec_of_operator(Window)

    with pytest.raises(WorkflowCompileError, match="at least one collect"):
        spec.plan_ports(
            "clip",
            spec.validate_params({"size": 2}),
            [("x", "hold", EntryLayout(), ())],
        )


@pytest.mark.parametrize(
    ("inputs", "field_path", "axis_id", "message"),
    [
        (
            [
                (
                    "x",
                    "collect",
                    EntryLayout((Axis("d", AXIS_KIND_DYNAMIC_NESTING),)),
                    (),
                )
            ],
            ("collect", "x"),
            "d",
            r"axis 'd' \(dynamic_nesting\) is not stationary.*same logical entity "
            r"or region.*cannot verify.*Collapse data-dependent",
        ),
        (
            [("x", "collect", EntryLayout((Axis("n", AXIS_KIND_SAMPLE),)), ())],
            ("collect", "x"),
            "n",
            r"axis 'n' \(sample\) is not stationary",
        ),
        (
            [("x", "collect", EntryLayout((N, Axis("old", AXIS_KIND_TIME))), ())],
            ("collect", "x"),
            "old",
            "axis 'old' is already a time axis",
        ),
        (
            [
                ("x", "collect", EntryLayout((N,)), ()),
                ("h", "hold", EntryLayout((N, C)), ()),
            ],
            ("hold", "h"),
            None,
            "hold 'h': axes .* are not a parent prefix of collect 'x'",
        ),
    ],
)
def test_window_rejects_unstable_or_already_temporal_layouts(
    inputs, field_path, axis_id, message
) -> None:
    spec = spec_of_operator(Window)

    with pytest.raises(OperatorInputError, match=message) as raised:
        spec.plan_ports("clip", spec.validate_params({"size": 2}), inputs)

    assert raised.value.field_path == field_path
    assert raised.value.axis_id == axis_id
    assert raised.value.step_path == ("$operators", "clip")


def test_tumbling_window_appends_t_with_fresh_root_context() -> None:
    operator = windowed({"x": EntryLayout(), "y": EntryLayout()}, size=3)
    first = object()

    assert operator.push(row(0, x=sample(first, 0), y=sample("y0", 0))) == []
    assert operator.push(row(1, x=absent(), y=sample("y1", 33))) == []
    emitted = operator.push(row(2, x=sample("third", 66), y=sample("y2", 66)))

    collected = emitted[0].ports["x"]
    assert collected.layout.axes == (T,)
    assert collected.children[()] == ((0,), (1,), (2,))
    assert collected.values == {(0,): first, (2,): "third"}
    assert collected.values[(0,)] is first
    assert collected.filtered == frozenset({(1,)})
    assert collected.metadata.temporal_at(()) is None
    assert collected.metadata.temporal_at((1,)) is None
    assert collected.metadata.temporal_at((2,)).media_coverage == at(66)
    assert collected.metadata.sample_at(()) == SampleContext("cam")
    assert emitted[0].causes == (key("cam", 0), key("cam", 1), key("cam", 2))


def test_size_three_window_emits_once_per_three_rows() -> None:
    operator = windowed({"x": EntryLayout()}, size=3)

    emitted = [operator.push(row(index, x=sample(index, index))) for index in range(7)]

    assert [len(items) for items in emitted] == [0, 0, 1, 0, 0, 1, 0]
    assert operator.counters.peak_retained == 3
    assert operator.finish("eof") == []
    assert operator.counters.partial_dropped == 1


def test_sliding_window_shares_overlap_and_emits_a_fresh_tail_once() -> None:
    operator = windowed({"x": EntryLayout()}, size=3, step=2, partial="emit")
    shared = object()

    operator.push(row(0, x=sample(0, 0)))
    operator.push(row(1, x=sample(1, 1)))
    first = operator.push(row(2, x=sample(shared, 2)))
    operator.push(row(3, x=sample(3, 3)))
    tail = operator.finish("eof")

    assert first[0].ports["x"].values[(2,)] is shared
    assert tail[0].ports["x"].values == {(0,): shared, (1,): 3}
    assert operator.finish("eof") == []


def test_overlap_only_rows_never_form_a_duplicate_tail() -> None:
    operator = windowed({"x": EntryLayout()}, size=3, step=1, partial="emit")

    emitted = [operator.push(row(index, x=sample(index, index))) for index in range(4)]

    assert [len(items) for items in emitted] == [0, 0, 1, 1]
    assert operator.finish("stop") == []
    assert operator.counters.partial_dropped == 0


def test_filtered_rows_are_skipped_and_hold_follows_the_last_contributing_row() -> None:
    operator = windowed(
        {"x": EntryLayout(), "y": EntryLayout()}, {"ref": EntryLayout()}, size=2
    )
    last_ref = sample("ref-1", 1)

    operator.push(row(0, x=sample("x0", 0), y=absent(), ref=sample("ref-0", 0)))
    operator.push(row(1, x=absent(), y=absent(), ref=sample("ignored", 1)))
    emitted = operator.push(row(2, x=absent(), y=sample("y2", 2), ref=last_ref))

    ports = emitted[0].ports
    assert ports["ref"] is last_ref
    assert ports["x"].values == {(0,): "x0"} and ports["x"].filtered == {(1,)}
    assert ports["y"].values == {(1,): "y2"} and ports["y"].filtered == {(0,)}
    assert operator.counters.dropped == 1
    assert emitted[0].causes == (key("cam", 0), key("cam", 2))


def test_stationary_crops_gain_t_without_compacting_holes() -> None:
    layout = EntryLayout((N, C))
    frames = EntryLayout((N,))
    operator = windowed({"crops": layout}, {"frame": frames}, size=2)
    children = {(): ((0,),), (0,): ((0, 0), (0, 1))}

    def frame(name: str) -> Entry:
        return grouped({(0,): name}, layout=frames, children={(): ((0,),)})

    operator.push(
        row(
            0,
            crops=grouped(
                {(0, 0): "c00", (0, 1): "c01"}, layout=layout, children=children
            ),
            frame=frame("f0"),
        )
    )
    emitted = operator.push(
        row(
            1,
            crops=grouped(
                {(0, 0): "c10"}, layout=layout, children=children, filtered={(0, 1)}
            ),
            frame=frame("f1"),
        )
    )

    assert emitted[0].ports["frame"].values == {(0,): "f1"}
    crops = emitted[0].ports["crops"]
    assert crops.layout.axes == (N, C, T)
    assert crops.children[(0,)] == ((0, 0), (0, 1))
    assert crops.children[(0, 1)] == ((0, 1, 0), (0, 1, 1))
    assert crops.values == {(0, 0, 0): "c00", (0, 0, 1): "c10", (0, 1, 0): "c01"}
    assert crops.filtered == {(0, 1, 1)}


def _camera_batch(
    identities: Sequence[Optional[str]], *, metadata: Any = None
) -> Entry:
    layout = EntryLayout((N,))
    positions = tuple((index,) for index in range(len(identities)))
    sample_map = {(): None}
    for index, identity in zip(positions, identities):
        if identity is not None:
            sample_map[index] = SampleContext(
                identity, "camera", source_metadata=metadata or {}
            )

    return grouped(
        {index: f"frame{index}" for index in positions},
        layout=layout,
        children={(): positions},
        sample=sample_map,
    )


def test_contradictory_parent_identity_fails_but_changing_metadata_does_not() -> None:
    operator = windowed({"frames": EntryLayout((N,))}, size=2)
    operator.push(
        row(0, frames=_camera_batch(["left", "right"], metadata={"frame": 1}))
    )
    emitted = operator.push(
        row(1, frames=_camera_batch(["left", "right"], metadata={"frame": 2}))
    )
    assert len(emitted) == 1

    operator.push(row(2, frames=_camera_batch(["left", "right"])))
    with pytest.raises(
        OperatorError,
        match=r"position \[0\] is 'left' \(camera\) in pulse cam#2 but 'right'",
    ):
        operator.push(row(3, frames=_camera_batch(["right", "left"])))


def test_unknown_identity_does_not_erase_a_known_one() -> None:
    operator = windowed({"frames": EntryLayout((N,))}, size=3)

    operator.push(row(0, frames=_camera_batch(["left"])))
    operator.push(row(1, frames=_camera_batch([None])))
    with pytest.raises(OperatorError, match="'left' .* but 'right'"):
        operator.push(row(2, frames=_camera_batch(["right"])))


def test_close_releases_rows_without_emitting() -> None:
    operator = windowed({"x": EntryLayout()}, size=3, partial="emit")
    operator.push(row(0, x=sample(0, 0)))

    operator.close()

    assert operator.finish("eof") == []
