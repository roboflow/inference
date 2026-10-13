"""``v2/window@v1`` history options, driven directly without a runtime.

Pulses come from a sparse set of stationary cameras: a camera absent from a
pulse has no member there. Payloads are short strings such as ``"c1@3"``
(camera 1, pulse 3), so a window reads as the history it holds.
"""

import gc
import weakref
from fractions import Fraction
from typing import Any, Dict, List, Optional, Sequence

import pytest
from roboflow_workflows.v2.data.batch import (
    AXIS_KIND_SAMPLE,
    AXIS_KIND_STATIC_NESTING,
    AXIS_KIND_TIME,
    Axis,
    EntryLayout,
    EntryMetadata,
    SampleContext,
    TemporalContext,
    Timestamp,
)
from roboflow_workflows.v2.data.kinds import Kind
from roboflow_workflows.v2.engine.errors import (
    OperatorError,
    OperatorInputError,
    ParamsValidationError,
    WorkflowCompileError,
)
from roboflow_workflows.v2.engine.execution.entries import Entry, scalar_entry
from roboflow_workflows.v2.engine.operators import (
    Arrival,
    OperatorInput,
    OperatorPulse,
    spec_of_operator,
)
from roboflow_workflows.v2.engine.operators.window import Window
from roboflow_workflows.v2.engine.plan import PulseKey

ITEM = Kind(name="item", description="A test payload.")
CAM = Axis("sources.cams:camera", AXIS_KIND_SAMPLE, stationary=True)
CROP = Axis("crop:regions", AXIS_KIND_STATIC_NESTING)
T = Axis("operators.history:t", AXIS_KIND_TIME)
CAMERAS = EntryLayout((CAM,))
CROPS = EntryLayout((CAM, CROP))


def history(
    collect: Dict[str, EntryLayout],
    hold: Optional[Dict[str, EntryLayout]] = None,
    **params: Any,
) -> Window:
    spec = spec_of_operator(Window)
    validated = spec.validate_params(params, operator_name="history")
    inputs = [
        OperatorInput(name, "collect", layout, (ITEM,))
        for name, layout in collect.items()
    ]
    inputs += [
        OperatorInput(name, "hold", layout, (ITEM,))
        for name, layout in (hold or {}).items()
    ]
    spec.plan_ports("history", validated, inputs)

    return Window(name="history", params=validated, inputs=inputs)


def pulse(sequence: int, **entries: Entry) -> List[Arrival]:
    return [
        Arrival(input=name, entry=entry, pulse=PulseKey("run", "cams", sequence))
        for name, entry in entries.items()
    ]


def observed(sequence: int) -> TemporalContext:
    return TemporalContext(observed_coverage=Timestamp(sequence, Fraction(1), "tick"))


def cameras(
    sequence: int, members: Dict[int, Any], *, sources: Optional[Dict[int, str]] = None
) -> Entry:
    """Present cameras of one pulse; each member has its own contexts."""
    positions = tuple((camera,) for camera in sorted(members))
    sources = sources or {}
    sample = {(): None}
    temporal = {(): None}
    for position in positions:
        source = sources.get(position[0], f"cam{position[0]}")
        sample[position] = SampleContext(source, "camera")
        temporal[position] = observed(sequence)

    return Entry(
        layout=CAMERAS,
        metadata=EntryMetadata(sample=sample, temporal=temporal),
        children={(): positions},
        values={position: members[position[0]] for position in positions},
        filtered=frozenset(),
    )


def frames(sequence: int, *present: int) -> Entry:
    return cameras(sequence, {camera: f"c{camera}@{sequence}" for camera in present})


def flags(values: Dict[int, bool]) -> Entry:
    return cameras(0, values)


def root_flag(value: bool) -> Entry:
    return scalar_entry(value, metadata=EntryMetadata())


def windows(emitted: Sequence[OperatorPulse], name: str = "frames") -> List[dict]:
    """Each emission as ``{camera: [payload or None per T position]}``."""
    rendered = []
    for item in emitted:
        entry = item.ports[name]
        rendered.append(
            {
                parent[0]: [entry.values.get(leaf) for leaf in entry.children[parent]]
                for parent in entry.children[()]
            }
        )

    return rendered


def drive(operator: Window, pulses: Sequence[Sequence[int]]) -> List[dict]:
    """Push one pulse per camera list; render the windows emitted."""
    emitted = []
    for sequence, present in enumerate(pulses):
        produced = operator.push(pulse(sequence, frames=frames(sequence, *present)))
        assert len(produced) <= 1
        emitted.extend(windows(produced))

    return emitted


SPARSE = [[0, 1], [0], [0], [1], [0, 1]]
"""Camera 1 is absent from pulses 1 and 2; batches are smaller than the set."""


# Parameters ------------------------------------------------------------------


def test_defaults_keep_the_tumbling_pulse_window() -> None:
    operator = history({"frames": CAMERAS}, size=2)

    assert operator.params.align == "pulse"
    assert operator.params.fill == "wait"
    assert operator.params.order == "oldest_first"
    assert operator.params.clear_when is None
    assert drive(operator, SPARSE) == [
        {0: ["c0@0", "c0@1"], 1: ["c1@0", None]},
        {0: ["c0@2", None], 1: [None, "c1@3"]},
    ]


def test_position_alignment_slides_by_one_when_step_is_omitted() -> None:
    omitted = history({"frames": CAMERAS}, size=2, align="position")
    explicit = history({"frames": CAMERAS}, size=2, align="position", step=1)

    assert drive(omitted, SPARSE) == drive(explicit, SPARSE)
    assert drive(history({"frames": CAMERAS}, size=2, align="position"), SPARSE) == [
        {0: ["c0@0", "c0@1"]},
        {0: ["c0@1", "c0@2"]},
        {1: ["c1@0", "c1@3"]},
        {0: ["c0@2", "c0@4"], 1: ["c1@3", "c1@4"]},
    ]


@pytest.mark.parametrize("step", [2, 3])
def test_position_alignment_rejects_another_step(step: int) -> None:
    spec = spec_of_operator(Window)

    with pytest.raises(ParamsValidationError, match="align='position'.*set it to 1"):
        spec.validate_params({"size": 3, "step": step, "align": "position"})


def test_clear_when_must_name_a_hold_input_the_alignment_can_clear() -> None:
    with pytest.raises(WorkflowCompileError, match="'reset' is not a hold input"):
        history(
            {"frames": CAMERAS}, {"other": EntryLayout()}, size=2, clear_when="reset"
        )
    with pytest.raises(WorkflowCompileError, match="align='pulse' keeps whole pulses"):
        history({"frames": CAMERAS}, {"reset": CAMERAS}, size=2, clear_when="reset")

    history({"frames": CAMERAS}, {"reset": EntryLayout()}, size=2, clear_when="reset")


def test_plan_ports_keep_one_innermost_time_axis() -> None:
    spec = spec_of_operator(Window)
    params = spec.validate_params({"size": 2, "align": "position", "fill": "pad"})

    ports = spec.plan_ports(
        "history",
        params,
        [("crops", "collect", CROPS, (ITEM,)), ("frames", "collect", CAMERAS, (ITEM,))],
    )

    assert ports["crops"].layout.axes == (CAM, CROP, T)
    assert ports["frames"].layout.axes == (CAM, T)


def test_unstable_input_axis_names_the_stationary_declaration() -> None:
    typed_root = EntryLayout((Axis("inputs", AXIS_KIND_SAMPLE),))

    with pytest.raises(OperatorInputError) as raised:
        history({"frames": typed_root}, size=2)

    message = str(raised.value)
    assert (
        "axis 'inputs' (sample) is not stationary; it comes from a workflow input"
        in message
    )
    assert (
        '{"name": "<input>", "kind": ["item"], "axes": [{"id": "camera", '
        '"kind": "sample", "stationary": true}]}'
    ) in message
    assert "Collapse" not in message


def test_unstable_crop_axis_keeps_the_collapse_guidance() -> None:
    crops = EntryLayout((CAM, Axis("crop:found", "dynamic_nesting")))

    with pytest.raises(OperatorInputError, match="Collapse data-dependent") as raised:
        history({"crops": crops}, size=2)

    assert 'without "type"' not in str(raised.value)


# Position alignment ----------------------------------------------------------


def test_absent_camera_keeps_its_rows_and_only_present_ready_cameras_emit() -> None:
    pulse_aligned = drive(history({"frames": CAMERAS}, size=3, step=1), SPARSE)
    by_position = drive(history({"frames": CAMERAS}, size=3, align="position"), SPARSE)

    assert by_position == [
        {0: ["c0@0", "c0@1", "c0@2"]},
        {0: ["c0@1", "c0@2", "c0@4"], 1: ["c1@0", "c1@3", "c1@4"]},
    ]
    assert pulse_aligned[-1][1] == [None, "c1@3", "c1@4"]


def test_grow_emits_from_the_first_row_then_full_windows() -> None:
    operator = history({"frames": CAMERAS}, size=2, align="position", fill="grow")

    assert drive(operator, SPARSE) == [
        {0: ["c0@0"], 1: ["c1@0"]},
        {0: ["c0@0", "c0@1"]},
        {0: ["c0@1", "c0@2"]},
        {1: ["c1@0", "c1@3"]},
        {0: ["c0@2", "c0@4"], 1: ["c1@3", "c1@4"]},
    ]
    assert operator.finish("eof") == []


@pytest.mark.parametrize(
    ("order", "expected"),
    [
        ("oldest_first", [[None, None, "c0@0"], [None, "c0@0", "c0@1"]]),
        ("newest_first", [["c0@0", None, None], ["c0@1", "c0@0", None]]),
    ],
)
def test_pad_fills_missing_rows_with_filtered_holes(order: str, expected) -> None:
    operator = history(
        {"frames": CAMERAS}, size=3, align="position", fill="pad", order=order
    )

    first = operator.push(pulse(0, frames=frames(0, 0)))
    second = operator.push(pulse(1, frames=frames(1, 0)))[0].ports["frames"]

    assert windows(first)[0][0] == expected[0]
    assert [second.values.get(leaf) for leaf in second.children[(0,)]] == expected[1]
    holes = {leaf for leaf in second.children[(0,)] if leaf not in second.values}
    assert holes == second.filtered
    for hole in holes:
        assert hole not in second.metadata.sample
        assert hole not in second.metadata.temporal


def test_newest_first_members_keep_their_own_contexts() -> None:
    operator = history(
        {"frames": CAMERAS}, size=2, align="position", order="newest_first"
    )
    operator.push(pulse(0, frames=frames(0, 0)))

    emitted = operator.push(pulse(1, frames=frames(1, 0)))[0]

    entry = emitted.ports["frames"]
    assert [entry.values[leaf] for leaf in entry.children[(0,)]] == ["c0@1", "c0@0"]
    assert entry.metadata.temporal_at((0, 0)) == observed(1)
    assert entry.metadata.temporal_at((0, 1)) == observed(0)
    assert entry.metadata.sample_at((0, 1)) == SampleContext("cam0", "camera")
    assert entry.metadata.sample_at((0,)) == SampleContext("cam0", "camera")
    assert entry.metadata.temporal_at(()) is None
    assert emitted.causes == (PulseKey("run", "cams", 0), PulseKey("run", "cams", 1))


def test_joined_cameras_keep_their_own_parent_contexts() -> None:
    operator = history({"frames": CAMERAS}, size=1, align="position")

    entry = operator.push(pulse(0, frames=frames(0, 0, 2)))[0].ports["frames"]

    assert entry.children[()] == ((0,), (2,))
    assert entry.metadata.sample_at((2,)) == SampleContext("cam2", "camera")
    assert entry.metadata.sample_at(()) is None
    assert entry.metadata.temporal_at((2, 0)) == observed(0)


@pytest.mark.parametrize("fill", ["grow", "pad"])
@pytest.mark.parametrize("order", ["oldest_first", "newest_first"])
@pytest.mark.parametrize("layout", [CAMERAS, CROPS], ids=["cameras", "crops"])
def test_sparse_collect_field_filters_only_its_track(fill, order, layout) -> None:
    operator = history(
        {"x": layout, "y": layout}, size=2, align="position", fill=fill, order=order
    )
    positions = [(0,), (1,)] if layout == CAMERAS else [(0, 0), (0, 1)]
    new, retained = positions

    def sparse(sequence: int, members: Dict[tuple, Any]) -> Entry:
        children = {}
        for index in members:
            for level in range(len(index)):
                children.setdefault(index[:level], set()).add(index[: level + 1])

        return Entry(
            layout=layout,
            children={node: tuple(sorted(nodes)) for node, nodes in children.items()},
            values=members,
            filtered=frozenset(),
            metadata=EntryMetadata(
                sample={index: SampleContext(f"cam{index[0]}") for index in members},
                temporal={index: observed(sequence) for index in members},
            ),
        )

    missing = Entry(
        layout=layout,
        children={},
        values={},
        filtered=frozenset({()}),
        metadata=EntryMetadata(),
    )
    operator.push(pulse(0, x=sparse(0, {retained: 10}), y=sparse(0, {retained: 100})))

    emitted = operator.push(pulse(1, x=sparse(1, {new: 20, retained: 21}), y=missing))
    joined = emitted[0].ports["y"]

    previous, hole = (0, 1) if order == "oldest_first" else (1, 0)
    leaf = retained + (previous,)
    assert joined.layout.axes == (*layout.axes, T)
    assert joined.nodes_at(layout.depth) == positions
    assert joined.children[retained] == (retained + (0,), retained + (1,))
    assert joined.filtered == {new, retained + (hole,)}
    assert joined.values == {leaf: 100} and joined.has_value(leaf)
    assert joined.is_filtered(new) and new not in joined.children
    assert not joined.is_filtered(retained)
    assert joined.metadata.temporal_at(leaf) == observed(0)
    assert joined.metadata.sample_at(leaf) == SampleContext(f"cam{retained[0]}")
    assert joined.metadata.sample_at(new) is None
    assert retained + (hole,) not in joined.metadata.temporal
    assert retained + (hole,) not in joined.metadata.sample

    # Once both histories contain only absent fields, each track stays filtered.
    emitted = operator.push(pulse(2, x=sparse(2, {new: 30, retained: 31}), y=missing))
    absent = emitted[0].ports["y"]
    assert absent.values == {}
    assert absent.filtered == set(positions)
    assert absent.nodes_at(layout.depth) == positions
    assert absent.is_effectively_filtered()
    assert absent.layout == joined.layout


def test_sparse_join_keeps_empty_groups_distinct_from_missing_tracks() -> None:
    operator = history(
        {"frames": CAMERAS, "crops": CROPS},
        size=2,
        align="position",
        fill="grow",
    )
    empty_crops = Entry(
        layout=CROPS,
        children={(): ((1,),), (1,): ()},
        values={},
        filtered=frozenset(),
        metadata=EntryMetadata(sample={(1,): SampleContext("cam1", "camera")}),
    )
    missing = Entry(
        layout=CROPS,
        children={},
        values={},
        filtered=frozenset({()}),
        metadata=EntryMetadata(),
    )
    operator.push(pulse(0, frames=frames(0, 1), crops=empty_crops))

    joined = operator.push(pulse(1, frames=frames(1, 0, 1), crops=missing))[0].ports[
        "crops"
    ]

    assert joined.children == {(): ((0,), (1,)), (1,): ()}
    assert joined.filtered == {(0,)}
    assert joined.is_effectively_filtered((0,))
    assert not joined.is_effectively_filtered((1,))
    assert joined.metadata.sample_at((1,)) == SampleContext("cam1", "camera")
    assert joined.metadata.sample_at((0,)) is None
    assert joined.layout.axes == (CAM, CROP, T)


def test_stationary_crops_keep_history_per_crop_at_depth_two() -> None:
    operator = history({"crops": CROPS}, size=2, align="position")

    def crops(sequence: int, present: Dict[int, Sequence[int]]) -> Entry:
        children = {(): tuple((camera,) for camera in sorted(present))}
        values = {}
        for camera, regions in present.items():
            children[(camera,)] = ((camera, 0), (camera, 1))
            values.update({(camera, r): f"c{camera}r{r}@{sequence}" for r in regions})
        filtered = {
            (camera, r)
            for camera in present
            for r in (0, 1)
            if (camera, r) not in values
        }
        return Entry(
            layout=CROPS,
            metadata=EntryMetadata(
                sample={
                    (camera,): SampleContext(f"cam{camera}", "camera")
                    for camera in present
                },
                temporal={(): observed(sequence)},
            ),
            children=children,
            values=values,
            filtered=frozenset(filtered),
        )

    operator.push(pulse(0, crops=crops(0, {0: [0, 1], 1: [0]})))
    first = operator.push(pulse(1, crops=crops(1, {0: [0]})))[0].ports["crops"]
    second = operator.push(pulse(2, crops=crops(2, {0: [1], 1: [0, 1]})))[0]

    assert first.layout.axes == (CAM, CROP, T)
    assert first.values == {(0, 0, 0): "c0r0@0", (0, 0, 1): "c0r0@1"}
    joined = second.ports["crops"]
    assert joined.children[()] == ((0,), (1,))
    assert joined.children[(0,)] == ((0, 1),)
    assert joined.children[(1,)] == ((1, 0),)
    assert joined.values == {
        (0, 1, 0): "c0r1@0",
        (0, 1, 1): "c0r1@2",
        (1, 0, 0): "c1r0@0",
        (1, 0, 1): "c1r0@2",
    }
    assert joined.metadata.sample_at((1, 0, 1)) == SampleContext("cam1", "camera")


def test_two_collect_layouts_align_at_their_shared_camera_axis() -> None:
    operator = history(
        {"frames": CAMERAS, "crops": CROPS},
        {"reset": CAMERAS},
        size=2,
        align="position",
        clear_when="reset",
    )

    def crops(sequence: int, *present: int) -> Entry:
        return Entry(
            layout=CROPS,
            metadata=EntryMetadata(
                sample={(c,): SampleContext(f"cam{c}", "camera") for c in present}
            ),
            children={
                (): tuple((c,) for c in present),
                **{(c,): ((c, 0),) for c in present},
            },
            values={(c, 0): f"c{c}r0@{sequence}" for c in present},
            filtered=frozenset(),
        )

    no_reset = flags({})
    operator.push(
        pulse(0, frames=frames(0, 0, 1), crops=crops(0, 0, 1), reset=no_reset)
    )
    emitted = operator.push(
        pulse(1, frames=frames(1, 0), crops=crops(1, 0), reset=no_reset)
    )[0]
    later = operator.push(
        pulse(2, frames=frames(2, 1), crops=crops(2), reset=no_reset)
    )[0]

    assert windows([emitted]) == [{0: ["c0@0", "c0@1"]}]
    assert emitted.ports["crops"].values == {(0, 0, 0): "c0r0@0", (0, 0, 1): "c0r0@1"}
    assert windows([later]) == [{1: ["c1@0", "c1@2"]}]
    crops_at_1 = later.ports["crops"]
    assert crops_at_1.values == {(1, 0, 0): "c1r0@0"}
    assert crops_at_1.filtered == {(1, 0, 1)}


# Clearing ----------------------------------------------------------------------


def test_clear_at_a_camera_resets_only_that_camera() -> None:
    operator = history(
        {"frames": CAMERAS},
        {"reset": CAMERAS},
        size=3,
        align="position",
        fill="grow",
        clear_when="reset",
    )

    def step(sequence: int, present: Sequence[int], reset: Dict[int, bool]):
        return windows(
            operator.push(
                pulse(sequence, frames=frames(sequence, *present), reset=flags(reset))
            )
        )

    step(0, [0, 1], {0: False, 1: False})
    step(1, [0, 1], {0: False, 1: False})

    assert step(2, [0, 1], {0: False, 1: True}) == [
        {0: ["c0@0", "c0@1", "c0@2"], 1: ["c1@2"]}
    ]
    assert step(3, [1], {1: False}) == [{1: ["c1@2", "c1@3"]}]


def test_root_clear_resets_every_camera_including_absent_ones() -> None:
    operator = history(
        {"frames": CAMERAS},
        {"reset": EntryLayout()},
        size=2,
        align="position",
        partial="emit",
        clear_when="reset",
    )
    operator.push(pulse(0, frames=frames(0, 0, 1), reset=root_flag(False)))

    assert operator.push(pulse(1, frames=frames(1, 0), reset=root_flag(True))) == []
    assert operator.counters.dropped == 2
    assert windows(operator.finish("eof")) == [{0: ["c0@1"]}]


def test_cleared_camera_keeps_its_source_identity() -> None:
    operator = history(
        {"frames": CAMERAS},
        {"reset": CAMERAS},
        size=2,
        align="position",
        clear_when="reset",
    )
    operator.push(pulse(0, frames=frames(0, 0), reset=flags({0: False})))
    operator.push(pulse(1, frames=frames(1, 0), reset=flags({0: True})))

    swapped = cameras(2, {0: "x"}, sources={0: "elsewhere"})
    with pytest.raises(
        OperatorError,
        match=r"position \[0\] is 'cam0' \(camera\) in pulse cams#0 but 'elsewhere'",
    ):
        operator.push(pulse(2, frames=swapped, reset=flags({0: False})))


def test_identity_contradiction_fails_across_slides() -> None:
    operator = history({"frames": CAMERAS}, size=1, align="position")
    for sequence in range(3):
        operator.push(pulse(sequence, frames=frames(sequence, 0)))

    with pytest.raises(OperatorError, match="'cam0' .* but 'cam9'"):
        operator.push(pulse(3, frames=cameras(3, {0: "x"}, sources={0: "cam9"})))


# Termination, accounting and retention ----------------------------------------


def test_finish_emits_each_position_unemitted_rows_once() -> None:
    operator = history({"frames": CAMERAS}, size=3, align="position", partial="emit")
    drive(operator, [[0, 1], [0], [0], [2]])

    tail = operator.finish("eof")

    assert windows(tail) == [{1: ["c1@0"], 2: ["c2@3"]}]
    assert tail[0].causes == (PulseKey("run", "cams", 0), PulseKey("run", "cams", 3))
    assert operator.finish("eof") == []


def test_dropped_tails_are_counted_per_position() -> None:
    operator = history({"frames": CAMERAS}, size=3, align="position")
    drive(operator, [[0, 1], [0], [0], [2]])

    assert operator.finish("stop") == []
    assert operator.counters.partial_dropped == 2


def test_at_most_one_emission_per_push_and_bounded_retention() -> None:
    operator = history({"frames": CAMERAS}, size=2, align="position", fill="grow")
    pulses = [[0, 1, 2], [1], [0, 2], [], [0, 1, 2], [2]] * 3

    emitted = []
    for sequence, present in enumerate(pulses):
        entry = frames(sequence, *present) if present else cameras(sequence, {})
        emitted.append(len(operator.push(pulse(sequence, frames=entry))))

    assert max(emitted) == 1
    assert sum(emitted) == sum(1 for present in pulses if present)
    assert operator.counters.dropped == 3
    assert operator.counters.peak_retained == 2 * 3


class Payload:
    """A weakly referenceable payload."""


def test_a_track_retains_only_its_own_position_payloads() -> None:
    operator = history(
        {"frames": CAMERAS},
        {"reset": CAMERAS},
        size=3,
        align="position",
        clear_when="reset",
    )
    first, second = Payload(), Payload()
    watched = weakref.ref(first)
    operator.push(pulse(0, frames=cameras(0, {0: first, 1: second}), reset=flags({})))
    operator.push(pulse(1, frames=frames(1, 1), reset=flags({0: True})))
    del first
    gc.collect()

    assert watched() is None
