"""Scheduling examples on the SYNTHETIC probe workload (``bounded_probes``).

Each example forces one interleaving with probe events, then checks what the
engine did: who overlapped, in which order results arrived, what the counters
say. Nothing here measures or claims speed.
"""

import copy
import json
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from bounded_probes import (
    PROBE_CATALOGUE,
    WAIT_SECONDS,
    Probe,
    ProbeTimeout,
    ProbeWholeCallModel,
    expect,
)
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import ActiveRunError
from roboflow_workflows.execution_engine.v2.pipelining import PipelineOptions
from roboflow_workflows.execution_engine.v2.pipelining.passive import (
    PipelineAbortedError,
)
from roboflow_workflows.execution_engine.v2.plan import CompiledWorkflow, CompileOptions

WORKFLOWS = Path(__file__).resolve().parent / "workflows"


def load_definition(name: str) -> Dict[str, Any]:
    """Read one workflow of this directory.

    Args:
        name: File name in ``workflows/``.

    Returns:
        The parsed definition.
    """
    definition = json.loads((WORKFLOWS / name).read_text())

    return definition


def compile_probe(definition: Mapping[str, Any]) -> CompiledWorkflow:
    """Compile a probe workflow in phase mode, so phases are separate stages.

    Args:
        definition: Workflow definition using ``probe/*`` blocks.

    Returns:
        The plan.
    """
    plan = compile_workflow(
        definition,
        catalogue=PROBE_CATALOGUE,
        options=CompileOptions(block_execution="phases"),
    )

    return plan


def field_value(result: Any, name: str) -> Any:
    """Value of a single-port output of a ``RunResult`` or ``GroupResult``."""
    (entry,) = result.selections[name].values()
    value = result.outputs.data[entry]

    return value


def timeline_records(probe: Probe) -> List[Dict[str, Any]]:
    """The probe timeline as JSON records."""
    records = [
        {"order": order, "event": event, "call": key, "run_id": run_id}
        for order, event, key, run_id in probe.timeline
    ]

    return records


def passive_timeline() -> Dict[str, Any]:
    """Run 1 enters phase ``first`` while run 0 is inside ``second``; opt-out does not.

    Returns:
        Evidence per ``phase_overlap`` setting and the serial reference.
    """
    evidence: Dict[str, Any] = {}
    for label, overlap in (
        ("phase_overlap=True", True),
        ("phase_overlap=False", False),
    ):
        definition = load_definition("passive_probe.json")
        if not overlap:
            definition["steps"][1]["type"] = ProbeWholeCallModel.type
        probe = Probe()
        probe.hold("second:p0")
        session = compile_probe(definition).create_session({"probe": probe})

        with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
            first = pipeline.submit({"value": 0.0, "tag": "p0"})
            probe.reached("second:p0")
            second = pipeline.submit({"value": 1.0, "tag": "p1"})
            if overlap:
                # Forced: run 0 stays inside "second" until run 1 entered "first".
                probe.reached("first:p1")
            probe.release("second:p0")
            scores = [
                field_value(future.result(timeout=WAIT_SECONDS), "score")
                for future in (first, second)
            ]

        expect(f"{label}: scores", scores, [0.0, 10.0])
        model_overlap = ("second:p0", "first:p1") in probe.overlaps()
        expect(f"{label}: run 1 in first while run 0 in second", model_overlap, overlap)
        evidence[label] = {
            "scores": scores,
            "overlapping_calls": probe.overlaps(),
            "timeline": timeline_records(probe),
            "counters": pipeline.counters.snapshot(),
            "outcomes": pipeline.outcomes,
        }

    serial = compile_probe(load_definition("passive_probe.json")).create_session(
        {"probe": Probe()}
    )
    reference = [
        field_value(serial.run({"value": value, "tag": f"s{index}"}), "score")
        for index, value in enumerate((0.0, 1.0))
    ]
    expect("serial reference scores", reference, [0.0, 10.0])
    evidence["serial_reference_scores"] = reference

    return evidence


class Deliveries:
    """Group handlers recording ``(group, tag)`` in delivery order.

    Args:
        groups: Group names to handle.
        expected: Deliveries per group after which ``complete(group)`` is set.
    """

    def __init__(self, groups: List[str], *, expected: Mapping[str, int]):
        self._lock = threading.Lock()
        self.order: List[Dict[str, Any]] = []
        self._expected = dict(expected)
        self._counts = dict.fromkeys(groups, 0)
        self._complete = {group: threading.Event() for group in groups}
        self.handlers: Dict[str, Callable[[Any], None]] = {
            group: self._handler(group) for group in groups
        }

    def wait_complete(self, group: str) -> None:
        """Wait until ``group`` received its expected deliveries.

        Raises:
            ProbeTimeout: When it does not happen within ``WAIT_SECONDS``.
        """
        if not self._complete[group].wait(WAIT_SECONDS):
            raise ProbeTimeout(f"group {group} was not completely delivered")

    def tags(self, group: Optional[str] = None) -> List[str]:
        """Delivered tags, of one group or all, in delivery order."""
        with self._lock:
            tags = [
                item["tag"] for item in self.order if group in (None, item["group"])
            ]

        return tags

    def _handler(self, group: str) -> Callable[[Any], None]:
        def handle(result: Any) -> None:
            with self._lock:
                self.order.append(
                    {
                        "group": group,
                        "tag": field_value(result, "tag"),
                        "source": result.pulse.source,
                        "sequence": result.pulse.sequence,
                    }
                )
                self._counts[group] += 1
                if self._counts[group] == self._expected.get(group, -1):
                    self._complete[group].set()

        return handle


def active_timeline() -> Dict[str, Any]:
    """Frames of one source overlap by phase and stay in order; sources are independent.

    Returns:
        Delivery order, overlapping calls, counters and the serial reference.
    """
    plan = compile_probe(load_definition("active_probe.json"))
    probe = Probe()
    probe.hold("second:a0")
    deliveries = Deliveries(["a", "b"], expected={"a": 4, "b": 2})
    run = plan.create_session({"probe": probe}).start(
        {},
        handlers=deliveries.handlers,
        admission_bound=2,
        pipeline=PipelineOptions(max_in_flight=3),
    )
    try:
        probe.reached("second:a0")
        probe.reached("first:a1")
        deliveries.wait_complete("b")
        # a0 is held, so no frame of a may be delivered yet (per-source order),
        # while b, an unrelated source, was delivered completely.
        expect("a delivered while a0 is held", deliveries.tags("a"), [])
    finally:
        probe.release("second:a0")
    expect("run completes", run.wait(timeout=WAIT_SECONDS), True)

    expect("a in frame order", deliveries.tags("a"), ["a0", "a1", "a2", "a3"])
    expect("b in frame order", deliveries.tags("b"), ["b0", "b1"])
    expect(
        "a1 entered first while a0 was inside second",
        ("second:a0", "first:a1") in probe.overlaps(),
        True,
    )

    serial = Deliveries(["a", "b"], expected={"a": 4, "b": 2})
    reference = plan.create_session({"probe": Probe()}).start(
        {}, handlers=serial.handlers, admission_bound=2
    )
    expect("serial run completes", reference.wait(timeout=WAIT_SECONDS), True)
    for group in ("a", "b"):
        expect(
            f"{group}: same results as serial",
            deliveries.tags(group),
            serial.tags(group),
        )

    evidence = {
        "delivery_order": deliveries.order,
        "serial_delivery_order": serial.order,
        "overlapping_calls": probe.overlaps(),
        "timeline": timeline_records(probe),
        "source_counters": _source_counters(run),
        "pipeline_counters": run.pipeline_counters.snapshot(),
    }

    return evidence


def overload() -> Dict[str, Any]:
    """A held consumer: ``block`` reads ahead by one; ``latest`` drops stale frames.

    Returns:
        Counters, delivered frames and result ages per policy.
    """
    evidence: Dict[str, Any] = {}
    for policy in ("block", "latest"):
        probe = Probe()
        # The reader waits inside read:f1 until f0 is inside the held consumer.
        probe.hold("read:f1", "slow:f0")
        deliveries = Deliveries(["frames"], expected={})
        run = (
            compile_probe(load_definition("overload_probe.json"))
            .create_session({"probe": probe})
            .start(
                {},
                handlers=deliveries.handlers,
                admission_bound=1,
                pipeline=PipelineOptions(max_in_flight=1, overload=policy),
            )
        )
        try:
            probe.reached("slow:f0")
            probe.release("read:f1")
            if policy == "latest":
                # The reader never waits for a slot: every frame is read
                # while f0 is still held.
                probe.reached("read:f9")
        finally:
            probe.release("read:f1", "slow:f0")
        expect(f"{policy}: run completes", run.wait(timeout=WAIT_SECONDS), True)

        counters = _source_counters(run)["camera"]
        delivered = deliveries.tags()
        expect(f"{policy}: every frame read", counters["read"], 10)
        expect(
            f"{policy}: read = admitted + dropped + unadmitted",
            counters["read"],
            counters["admitted"] + counters["dropped"] + counters["unadmitted"],
        )
        expect(
            f"{policy}: admitted = processed + cancelled",
            counters["admitted"],
            counters["processed"] + counters["cancelled"],
        )
        if policy == "block":
            expect("block: lossless", delivered, [f"f{n}" for n in range(10)])
            expect("block: nothing dropped", counters["dropped"], 0)
            order = {(event, key): n for n, event, key, _ in probe.timeline}
            expect(
                "block: f1 waits for f0's slot, so f2 is read after f0 left",
                order[("enter", "read:f2")] > order[("leave", "slow:f0")],
                True,
            )
        else:
            expect("latest: the held frame and the newest", delivered, ["f0", "f9"])
            expect("latest: dropped the stale frames", counters["dropped"], 8)
        snapshot = run.pipeline_counters.snapshot()
        evidence[policy] = {
            "delivered": delivered,
            "source_counters": counters,
            "result_age_ns": snapshot["result_age_ns"],
            "peaks": snapshot["peak"],
        }

    return evidence


def lifecycle() -> Dict[str, Any]:
    """Stop drains, cancel discards unstarted work, a failure is attributed; no thread survives.

    Returns:
        Final state, counters and surviving threads per scenario.
    """
    evidence: Dict[str, Any] = {}
    for scenario in ("stop", "cancel", "failure"):
        definition = load_definition("overload_probe.json")
        if scenario == "failure":
            definition["steps"][0]["fail_on"] = "f2"
        probe = Probe()
        probe.hold("slow:f1")
        deliveries = Deliveries(["frames"], expected={})
        before = set(threading.enumerate())
        run = (
            compile_probe(definition)
            .create_session({"probe": probe})
            .start(
                {},
                handlers=deliveries.handlers,
                admission_bound=2,
                pipeline=PipelineOptions(max_in_flight=2),
            )
        )
        failure: Optional[str] = None
        try:
            probe.reached("slow:f1")
            if scenario == "stop":
                run.stop()
            elif scenario == "cancel":
                run.cancel()
        finally:
            probe.release("slow:f1")
        try:
            expect(f"{scenario}: run ends", run.wait(timeout=WAIT_SECONDS), True)
        except ActiveRunError as error:
            failure = str(error)
        survivors = [
            thread.name
            for thread in set(threading.enumerate()) - before
            if thread.is_alive()
        ]
        counters = _source_counters(run)["camera"]
        expect(f"{scenario}: no run thread survives", survivors, [])
        expect(
            f"{scenario}: admitted = processed + cancelled",
            counters["admitted"],
            counters["processed"] + counters["cancelled"],
        )
        expect(
            f"{scenario}: read = admitted + dropped + unadmitted",
            counters["read"],
            counters["admitted"] + counters["dropped"] + counters["unadmitted"],
        )
        if scenario == "stop":
            expect("stop: admitted frames are delivered", counters["cancelled"], 0)
        if scenario == "failure":
            expect("failure: attributed", failure is not None and "f2" in failure, True)
        evidence[scenario] = {
            "state": run.state,
            "failure": failure,
            "delivered": deliveries.tags(),
            "source_counters": counters,
            "surviving_threads": survivors,
        }

    evidence["passive_failure"] = _passive_failure()

    return evidence


def _passive_failure() -> Dict[str, Any]:
    """A failing submission aborts the one waiting behind it; the session recovers."""
    definition = load_definition("passive_probe.json")
    definition["steps"][0]["fail_on"] = "p1"
    probe = Probe()
    probe.hold("load:p0")
    session = compile_probe(definition).create_session({"probe": probe})

    with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
        futures = [
            pipeline.submit({"value": float(number), "tag": f"p{number}"})
            for number in range(3)
        ]
        probe.release("load:p0")
        outcomes = []
        for future in futures:
            error = future.exception(timeout=WAIT_SECONDS)
            outcomes.append("ok" if error is None else type(error).__name__)

    expect(
        "failure: p1 fails, p2 aborted",
        outcomes[1:],
        ["StepExecutionError", "PipelineAbortedError"],
    )
    recovered = field_value(session.run({"value": 5.0, "tag": "after"}), "score")
    expect("the session runs again after close", recovered, 50.0)
    evidence = {
        "submission_outcomes": outcomes,
        "pipeline_outcomes": pipeline.outcomes,
        "failure": str(pipeline.failure),
        "aborted_error_type": PipelineAbortedError.__name__,
    }

    return evidence


def _source_counters(run: Any) -> Dict[str, Dict[str, Any]]:
    counters = {name: copy.copy(vars(source)) for name, source in run.counters.items()}

    return counters
