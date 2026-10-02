"""Named, self-checking examples of alignment, windows and temporal blocks.

Every case compiles authored JSON with the real V2 compiler, runs it on the
real active engine and asserts what it observed. Expected failures assert the
concrete error or counter; an unexpected success fails the case.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from blocks import GreyCard
from fixtures import SCHEDULES, make_image, media_timestamp
from gallery import Gallery
from host import (
    SENSOR_CSV,
    RunRecord,
    create_demo_catalogue,
    load_definition,
    run_active,
    with_changes,
)
from inspection import axis_kinds, leaf_pts, observe_fields
from probes import ReadOrder
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.data import (
    EntryMetadata,
    InputValue,
    SampleContext,
    TemporalContext,
)
from roboflow_workflows.execution_engine.v2.errors import WorkflowCompileError
from roboflow_workflows.execution_engine.v2.introspection import describe_workflow


@dataclass(frozen=True)
class Case:
    """One runnable example.

    Attributes:
        description: What the case shows, printed by ``--list``.
        run: Callable receiving the case output directory (or None) and
            returning JSON-friendly evidence.
    """

    description: str
    run: Callable[[Optional[Path]], Dict[str, Any]]


CASES: Dict[str, Case] = {}


def _case(name: str, description: str):
    def register(function):
        CASES[name] = Case(description=description, run=function)
        return function

    return register


def _write(directory: Optional[Path], name: str, document: Any) -> None:
    if directory is None:
        return

    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(document, indent=2, default=str) + "\n")


def _save_run(directory: Optional[Path], record: RunRecord) -> None:
    _write(directory, "compiled.json", describe_workflow(record.plan))
    _write(directory, "results.json", record.observations)


def _operator_counters(record: RunRecord) -> Dict[str, Dict[str, Any]]:
    counters = {
        name: dict(vars(counts))
        for name, counts in record.active.operator_counters.items()
    }

    return counters


def _source_counters(record: RunRecord) -> Dict[str, Dict[str, Any]]:
    counters = {
        name: dict(vars(counts)) for name, counts in record.active.counters.items()
    }

    return counters


def _error_record(error: BaseException) -> Dict[str, Any]:
    record = {
        "type": type(error).__name__,
        "message": str(error),
        "stage": getattr(error, "stage", None),
        "operator": getattr(error, "operator", None),
        "step_path": getattr(error, "step_path", None),
        "pulse": getattr(error, "pulse", None),
    }

    return record


def _compile_error(name: str, *fragments: str) -> Dict[str, Any]:
    definition = load_definition(name)
    try:
        compile_workflow(definition, catalogue=create_demo_catalogue())
    except WorkflowCompileError as error:
        message = str(error)
        missing = [fragment for fragment in fragments if fragment not in message]
        assert not missing, f"compile error lacks {missing}: {message}"
        print(f"  rejected as expected: {type(error).__name__}: {message}")
        evidence = {"file": name, "type": type(error).__name__, "message": message}
        return evidence

    raise AssertionError(f"{name} compiled, but it must be rejected")


def _pairs(record: RunRecord, group: str = "pairs") -> List[List[Any]]:
    pairs = [leaf_pts(observation, "samples") for observation in record.groups(group)]

    return pairs


# --- Main graph ---------------------------------------------------------------


@_case(
    "rig",
    "Main graph: two cameras + sensor, alignment, unaligned branch, windows, "
    "crops before/after T, best frame, last-PTS mosaic and recollection.",
)
def rig(directory: Optional[Path]) -> Dict[str, Any]:
    gallery = Gallery(
        None if directory is None else directory / "gallery",
        fields=[
            ("clips", "frames"),
            ("clips", "best"),
            ("clips", "half_strips"),
            ("clips", "clip_mosaic"),
            ("recollected", "summaries"),
        ],
    )
    record = run_active(
        load_definition("rig.json"),
        inputs={"sensor_csv": str(SENSOR_CSV)},
        on_result=gallery.collect,
    )
    _save_run(directory, record)

    # Named mixed fields without T: the sensor leads; 100 ms ties between
    # frames 80 and 120 and takes the earlier one.
    paired = record.groups("paired")
    assert [leaf_pts(item, "image") for item in paired] == [[0], [80], [200]]
    assert [item["fields"]["celsius"]["leaves"][0]["value"] for item in paired] == [
        21.5,
        22.0,
        23.5,
    ]
    for item in paired:
        assert axis_kinds(item, "image") == [] and axis_kinds(item, "celsius") == []

    # Batch without T: one stationary sample axis, left then right.
    samples = record.groups("rig_samples")
    assert [leaf_pts(item, "samples") for item in samples] == [
        [40 * i, 40 * i + 5] for i in range(8)
    ]
    for item in samples:
        assert axis_kinds(item, "samples") == ["sample"]
        assert axis_kinds(item, "halves") == ["sample", "static_nesting"]
        assert item["fields"]["samples"]["root_pts_ms"] is None

    # The unaligned sensor branch still runs once per reading.
    fahrenheit = [
        round(item["fields"]["fahrenheit"]["leaves"][0]["value"], 2)
        for item in record.groups("temperatures")
    ]
    assert fahrenheit == [70.7, 71.6, 74.3]

    clips = record.groups("clips")
    assert len(clips) == 2
    expected_best = [[80, 45], [240, 165]]
    expected_last = [[120, 125], [280, 285]]
    for clip, best, last in zip(clips, expected_best, expected_last):
        assert axis_kinds(clip, "frames") == ["sample", "time"]
        assert axis_kinds(clip, "crops") == ["sample", "static_nesting", "time"]
        assert axis_kinds(clip, "marker_crops") == ["sample", "time", "dynamic_nesting"]
        assert axis_kinds(clip, "best") == ["sample"]
        assert axis_kinds(clip, "half_strips") == ["sample", "static_nesting"]
        assert axis_kinds(clip, "clip_mosaic") == ["sample"]
        # No window-level timestamp is inherited from one member.
        assert clip["fields"]["frames"]["root_pts_ms"] is None
        # Different cameras choose different times; PTS is the chosen sample.
        assert leaf_pts(clip, "best") == best
        assert leaf_pts(clip, "reference") == last
        assert leaf_pts(clip, "clip_mosaic") == last
        assert leaf_pts(clip, "half_strips") == [last[0], last[0], last[1], last[1]]
        assert len(clip["causes"]) == 4

    (recollected,) = record.groups("recollected")
    assert axis_kinds(recollected, "summaries") == ["sample", "time"]
    assert leaf_pts(recollected, "summaries") == [120, 280, 125, 285]
    assert recollected["fields"]["summaries"]["root_pts_ms"] is None

    operators = _operator_counters(record)
    assert operators["rig"]["emitted"] == 8
    assert operators["paired"]["emitted"] == 3
    assert operators["clip"]["emitted"] == 2
    assert operators["recollect"]["emitted"] == 1
    page = gallery.write_index()
    evidence = {
        "paired_image_pts": [leaf_pts(item, "image") for item in paired],
        "best_pts": [leaf_pts(clip, "best") for clip in clips],
        "mosaic_last_pts": [leaf_pts(clip, "clip_mosaic") for clip in clips],
        "recollected_pts": leaf_pts(recollected, "summaries"),
        "layouts": {name: axis_kinds(clips[0], name) for name in clips[0]["fields"]},
        "operators": operators,
        "sources": _source_counters(record),
        "gallery": None if page is None else str(page),
    }

    return evidence


# --- Window termination and first/last/selected PTS ---------------------------


def _clip_windows(record: RunRecord) -> List[List[Any]]:
    windows = [leaf_pts(clip, "frames") for clip in record.groups("clips")]

    return windows


@_case("eof-drop", "Size-3 windows over 8 pairs: the 2-pair tail is dropped at EOF.")
def eof_drop(directory: Optional[Path]) -> Dict[str, Any]:
    record = run_active(load_definition("cameras.json"))
    _save_run(directory, record)
    assert _clip_windows(record) == [
        [0, 40, 80, 5, 45, 85],
        [120, 160, 200, 125, 165, 205],
    ]
    clip = _operator_counters(record)["clip"]
    assert clip["partial_dropped"] == 1 and clip["emitted"] == 2
    evidence = {"windows": _clip_windows(record), "clip": clip}

    return evidence


@_case("eof-partial", "Same run with partial='emit': one short final window.")
def eof_partial(directory: Optional[Path]) -> Dict[str, Any]:
    definition = with_changes(load_definition("cameras.json"), clip={"partial": "emit"})
    record = run_active(definition)
    _save_run(directory, record)
    windows = _clip_windows(record)
    assert windows[-1] == [240, 280, 245, 285]
    tail = record.groups("clips")[-1]
    assert [leaf["value"] for leaf in tail["fields"]["count"]["leaves"]] == [2, 2]
    assert leaf_pts(tail, "first") == [240, 245]
    assert leaf_pts(tail, "last") == [280, 285]
    clip = _operator_counters(record)["clip"]
    assert clip["partial_dropped"] == 0 and clip["emitted"] == 3
    evidence = {"windows": windows, "clip": clip}

    return evidence


@_case(
    "policies",
    "One [N,T] group, several outputs: first, last, selected and common_or_none PTS.",
)
def policies(directory: Optional[Path]) -> Dict[str, Any]:
    record = run_active(load_definition("cameras.json"))
    _save_run(directory, record)
    rows = []
    for clip in record.groups("clips"):
        row = {
            name: leaf_pts(clip, name) for name in ("first", "last", "best", "count")
        }
        rows.append(row)
        print(f"  clip {clip['pulse']}: {row}")
    assert rows == [
        {"first": [0, 5], "last": [80, 85], "best": [80, 45], "count": [None, None]},
        {
            "first": [120, 125],
            "last": [200, 205],
            "best": [200, 165],
            "count": [None, None],
        },
    ]
    best = record.groups("clips")[0]["fields"]["best"]["leaves"]
    # The selected payload is the member frame itself, not a copy.
    assert [leaf["value"]["image_id"] for leaf in best] == ["left@80ms", "right@45ms"]
    evidence = {"rows": rows}

    return evidence


@_case(
    "selection-k1",
    "Top-1 brightest frame stays a [N,K] collection; recollecting it is rejected.",
)
def selection_k1(directory: Optional[Path]) -> Dict[str, Any]:
    record = run_active(load_definition("selection.json"))
    _save_run(directory, record)
    rows = []
    for clip in record.groups("clips"):
        # K keeps its axis with one child per camera: (n, 0).
        assert axis_kinds(clip, "brightest") == ["sample", "dynamic_nesting"]
        leaves = clip["fields"]["brightest"]["leaves"]
        assert [leaf["index"] for leaf in leaves] == [[0, 0], [1, 0]]
        # A mosaic over K is a singular reduction back to [N].
        assert axis_kinds(clip, "brightest_tile") == ["sample"]
        rows.append(leaf_pts(clip, "brightest"))
    assert rows == [[120, 5], [280, 165]]
    rejection = _compile_error(
        "invalid/recollect_selection.json", "$steps.brightest.images"
    )
    evidence = {"brightest_pts": rows, "recollect_rejected": rejection}

    return evidence


# --- Alignment policies and errors --------------------------------------------


@_case("missing-drop", "Right camera misses its 85 ms frame; missing='drop' skips it.")
def missing_drop(directory: Optional[Path]) -> Dict[str, Any]:
    definition = with_changes(
        load_definition("cameras.json"), right={"schedule": "right-gap"}
    )
    record = run_active(definition)
    _save_run(directory, record)
    pairs = _pairs(record)
    assert [pair[0] for pair in pairs] == [0, 40, 120, 160, 200, 240, 280]
    rig_counts = _operator_counters(record)["rig"]
    assert rig_counts["dropped"] == 1 and rig_counts["emitted"] == 7
    evidence = {"pairs": pairs, "rig": rig_counts}

    return evidence


@_case(
    "missing-partial",
    "Same gap with missing='partial': the pair is emitted, right stays filtered.",
)
def missing_partial(directory: Optional[Path]) -> Dict[str, Any]:
    definition = with_changes(
        load_definition("cameras.json"),
        right={"schedule": "right-gap"},
        rig={"missing": "partial"},
    )
    record = run_active(definition)
    _save_run(directory, record)
    pairs = record.groups("pairs")
    assert len(pairs) == 8
    gap = pairs[2]["fields"]["samples"]
    assert leaf_pts(pairs[2], "samples") == [80]
    assert gap["filtered"] == [[1]]
    evidence = {"pairs": _pairs(record), "gap": gap}

    return evidence


@_case("late", "Right emits 60 ms after 85 ms; the late sample is counted, not used.")
def late(directory: Optional[Path]) -> Dict[str, Any]:
    definition = with_changes(
        load_definition("cameras.json"), right={"schedule": "right-late"}
    )
    record = run_active(definition)
    _save_run(directory, record)
    pairs = _pairs(record)
    assert pairs == [[40 * i, 40 * i + 5] for i in range(8)]
    rig_counts = _operator_counters(record)["rig"]
    assert rig_counts["late"] == 1
    evidence = {"pairs": pairs, "rig": rig_counts}

    return evidence


@_case(
    "clock-error", "Right camera uses another media clock: the run fails, attributed."
)
def clock_error(directory: Optional[Path]) -> Dict[str, Any]:
    definition = with_changes(
        load_definition("cameras.json"), right={"schedule": "right-other-clock"}
    )
    record = run_active(definition, expect_failure=True)
    error = _error_record(record.error)
    assert error["stage"] == "operator" and error["operator"] == "rig"
    assert "other-media" in error["message"] and "right" in error["message"]
    assert record.groups("pairs") == []
    assert all(counts["closed"] for counts in _source_counters(record).values())
    evidence = {"error": error, "sources": _source_counters(record)}
    _write(directory, "error.json", evidence)

    return evidence


@_case(
    "capacity",
    "Follower reads everything before the leader: enough max_pending succeeds, "
    "max_pending=4 fails explicitly.",
)
def capacity(directory: Optional[Path]) -> Dict[str, Any]:
    probed = with_changes(
        load_definition("cameras.json"),
        left={"type": "temporal_demo/leader_after_follower"},
        right={"type": "temporal_demo/follower_first"},
    )
    enough = run_active(probed, resources={"read_order": ReadOrder()})
    assert _pairs(enough) == [[40 * i, 40 * i + 5] for i in range(8)]

    tight = with_changes(probed, rig={"max_pending": 4})
    record = run_active(
        tight, resources={"read_order": ReadOrder()}, expect_failure=True
    )
    error = _error_record(record.error)
    assert error["stage"] == "operator" and error["operator"] == "rig"
    assert "right" in error["message"] and "4" in error["message"]
    assert record.groups("pairs") == []
    evidence = {"default_bound_pairs": _pairs(enough), "bound_4_error": error}
    _write(directory, "evidence.json", evidence)

    return evidence


@_case(
    "admission-one",
    "Window size 3 with admission_bound=1: retention is the operator's, not the source's.",
)
def admission_one(directory: Optional[Path]) -> Dict[str, Any]:
    record = run_active(load_definition("cameras.json"), admission_bound=1)
    _save_run(directory, record)
    assert len(record.groups("clips")) == 2
    sources = _source_counters(record)
    for counts in sources.values():
        assert counts["admitted"] == counts["processed"] == 8
        assert counts["cancelled"] == 0
    evidence = {"windows": _clip_windows(record), "sources": sources}

    return evidence


# --- Nested and gated paths ---------------------------------------------------


@_case(
    "nested",
    "Gated child workflows before the align operator and after the window.",
)
def nested(directory: Optional[Path]) -> Dict[str, Any]:
    record = run_active(load_definition("nested.json"))
    _save_run(directory, record)
    statuses = [
        item["fields"]["prepared"]["status"] for item in record.groups("left_frames")
    ]
    # Left frames at 0 and 160 ms are darker than 50 and never enter the child.
    assert statuses == [
        "filtered",
        "complete",
        "complete",
        "complete",
        "filtered",
        "complete",
        "complete",
        "complete",
    ]
    clips = record.groups("clips")
    assert [leaf_pts(clip, "frames") for clip in clips] == [
        [40, 80, 120, 45, 85, 125],
        [200, 240, 280, 205, 245, 285],
    ]
    for clip in clips:
        frames = clip["fields"]["frames"]["leaves"]
        assert all(leaf["value"]["shape"] == [3, 24, 32] for leaf in frames)
        # The last left frame is bright, the last right frame is dark: the
        # after-window child runs for left only.
        finished = clip["fields"]["finished"]
        assert finished["filtered"] == [[1]]
        assert leaf_pts(clip, "finished") == leaf_pts(clip, "last")[:1]
    evidence = {
        "left_statuses": statuses,
        "finished": [clip["fields"]["finished"] for clip in clips],
        "operators": _operator_counters(record),
    }

    return evidence


# --- Stop, failure and restart ------------------------------------------------


@_case("stop", "A handler stops the run on the first pair; admitted work drains.")
def stop(directory: Optional[Path]) -> Dict[str, Any]:
    def stop_on_first_pair(result, session) -> None:
        if result.group == "pairs":
            session.stop()

    record = run_active(load_definition("cameras.json"), on_result=stop_on_first_pair)
    _save_run(directory, record)
    assert record.active.state == "finished"
    sources = _source_counters(record)
    for counts in sources.values():
        assert counts["admitted"] == counts["processed"]
        assert counts["cancelled"] == 0 and counts["closed"]
    operators = _operator_counters(record)
    for counts in operators.values():
        assert counts["finished"] and counts["closed"]
    # Drained pairs are fewer than the full 8; a short tail is dropped.
    assert 1 <= len(record.groups("pairs")) < 8
    evidence = {
        "pairs": _pairs(record),
        "windows": _clip_windows(record),
        "sources": sources,
        "operators": operators,
    }

    return evidence


@_case(
    "failure",
    "A step fails inside the second window's pulse: attribution and cleanup.",
)
def failure(directory: Optional[Path]) -> Dict[str, Any]:
    record = run_active(load_definition("guarded_clips.json"), expect_failure=True)
    error = _error_record(record.error)
    assert error["stage"] == "step" and error["operator"] == "clip"
    assert error["pulse"] == 1
    assert "underexposed" in error["message"] and "guard" in error["message"]
    # The first window was delivered; nothing after the failure, not even the
    # partial tail that partial='emit' would otherwise produce.
    assert len(record.groups("clips")) == 1
    operators = _operator_counters(record)
    assert all(counts["closed"] for counts in operators.values())
    assert all(counts["closed"] for counts in _source_counters(record).values())
    evidence = {"error": error, "operators": operators}
    _write(directory, "error.json", evidence)

    return evidence


@_case("restart", "The same session starts twice; operator state starts fresh.")
def restart(directory: Optional[Path]) -> Dict[str, Any]:
    definition = load_definition("cameras.json")
    plan = compile_workflow(definition, catalogue=create_demo_catalogue())
    session = plan.create_session()
    first = run_active(definition, session=session)
    second = run_active(definition, session=session)
    assert first.observations and len(first.observations) == len(second.observations)
    assert _clip_windows(first) == _clip_windows(second)
    # Outcome counters restart from zero. peak_retained is left out: it depends
    # on how the two reader threads happen to interleave.
    outcomes = [
        {
            name: {
                key: value for key, value in counts.items() if key != "peak_retained"
            }
            for name, counts in _operator_counters(run).items()
        }
        for run in (first, second)
    ]
    assert outcomes[0] == outcomes[1]
    assert outcomes[1]["clip"]["emitted"] == 2
    evidence = {
        "windows": _clip_windows(second),
        "operators": _operator_counters(second),
    }

    return evidence


# --- Definitions the compiler must reject -------------------------------------


@_case(
    "invalid-dynamic-crop",
    "Window over dynamic crops: equal crop counts do not prove stable identity.",
)
def invalid_dynamic_crop(directory: Optional[Path]) -> Dict[str, Any]:
    evidence = _compile_error("invalid/dynamic_crop_window.json", "$steps.crop.crops")
    _write(directory, "error.json", evidence)

    return evidence


@_case("invalid-second-t", "Window over frames that already have a T axis.")
def invalid_second_t(directory: Optional[Path]) -> Dict[str, Any]:
    evidence = _compile_error("invalid/second_time_axis.json", "$operators.clip.frames")
    _write(directory, "error.json", evidence)

    return evidence


@_case(
    "invalid-middle-t",
    "Best frame over [N,T,C] crops: a T-oriented collapse needs T last.",
)
def invalid_middle_t(directory: Optional[Path]) -> Dict[str, Any]:
    evidence = _compile_error(
        "invalid/middle_time_collapse.json",
        "$steps.best",
        "markers:regions",
        "final time axis",
    )
    _write(directory, "error.json", evidence)

    return evidence


@_case(
    "invalid-lineage",
    "Best frame compares a window with the current rig pulse: equal [N] shapes, "
    "different pulses.",
)
def invalid_lineage(directory: Optional[Path]) -> Dict[str, Any]:
    evidence = _compile_error(
        "invalid/reference_from_other_pulse.json", "$steps.best", "rig", "clip"
    )
    _write(directory, "error.json", evidence)

    return evidence


@_case(
    "operator-in-child",
    "Operators are root declarations in this delivery; a child operator is rejected.",
)
def operator_in_child(directory: Optional[Path]) -> Dict[str, Any]:
    evidence = _compile_error("invalid/operator_in_child.json", "operators")
    _write(directory, "error.json", evidence)

    return evidence


# --- Passive (session.run) temporal input -------------------------------------


def _recorded_window() -> Dict[str, Any]:
    # A host that already holds a recorded [camera, time] window passes it in
    # with its timestamps; no operator runs and no state spans run() calls.
    frames, temporal, sample = [], {}, {}
    for n, name in enumerate(("left", "right")):
        schedule = SCHEDULES[name]
        frames.append(
            [
                make_image(
                    frame, schedule=schedule, image_id=f"{name}@{frame.pts_ms}ms"
                )
                for frame in schedule.frames[:4]
            ]
        )
        sample[(n,)] = SampleContext(source_id=name, source_type="recording")
        for t, frame in enumerate(schedule.frames[:4]):
            stamp = media_timestamp(frame.pts_ms)
            temporal[(n, t)] = TemporalContext(
                observed_coverage=stamp, media_coverage=stamp
            )
    references = [GreyCard().run(image=row[0], value=128)["image"] for row in frames]
    inputs = {
        "frames": InputValue(
            data=frames, metadata=EntryMetadata(sample=sample, temporal=temporal)
        ),
        "reference": InputValue(data=references, metadata=EntryMetadata(sample=sample)),
    }

    return inputs


@_case(
    "passive",
    "session.run with an explicit timestamped [N,T] input: temporal blocks work "
    "without operators; operators in a passive definition are rejected.",
)
def passive(directory: Optional[Path]) -> Dict[str, Any]:
    plan = compile_workflow(
        load_definition("passive.json"), catalogue=create_demo_catalogue()
    )
    session = plan.create_session()
    fields = observe_fields(session.run(_recorded_window()))
    again = observe_fields(session.run(_recorded_window()))
    _write(directory, "compiled.json", describe_workflow(plan))
    _write(directory, "results.json", fields)
    observed = {"fields": fields}
    assert leaf_pts(observed, "best") == [80, 45]
    assert leaf_pts(observed, "first") == [0, 5]
    assert leaf_pts(observed, "last") == [120, 125]
    assert leaf_pts(observed, "strip") == [120, 125]
    assert axis_kinds(observed, "best") == ["sample"]
    chosen = [leaf["value"]["image_id"] for leaf in fields["best"]["leaves"]]
    assert chosen == ["left@80ms", "right@45ms"]
    # Each run() is independent: the same window gives the same answers. The
    # mosaic canvas is a new image each time, so compare timestamps.
    repeated = {"fields": again}
    for name in fields:
        assert leaf_pts(repeated, name) == leaf_pts(observed, name)
    rejection = _compile_error("invalid/passive_operator.json", "operator")
    evidence = {"fields": fields, "operator_rejected": rejection}

    return evidence
