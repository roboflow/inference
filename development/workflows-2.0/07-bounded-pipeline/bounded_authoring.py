"""Authoring blocks for pipelined runs: a scratch-on-``self`` trap and two fixes.

With ``CompileOptions(block_execution="phases")`` and a pipeline, two pulses
may be in different phases of the SAME block instance at once. A phase never
runs twice at once, but ``self`` is shared by all phases::

    pulse 0:  prepare(0) ............ finish(0) reads self.scratch
    pulse 1:              prepare(1) writes self.scratch = 1

``ScratchOnSelf`` is an INTENTIONALLY UNSAFE block, not an engine defect: it
keeps scratch on ``self`` between phases. Forced into the interleaving above,
it returns ``[10, 10]`` instead of ``[0, 10]``. The fixes:

- ``ScratchAsOutput`` passes the scratch as a phase return value. The same
  forced interleaving gives the correct ``[0, 10]``.
- ``ScratchOnSelfWholeCall`` keeps the unsafe code but sets
  ``phase_overlap = False``: the whole call holds one stage, so pulse 1 cannot
  prepare before pulse 0 finished.

Every interleaving is forced with probe events (``bounded_probes.Probe``);
nothing sleeps and nothing is timed. No model runs.
"""

from typing import Any, Dict, List

from bounded_probes import Probe, expect
from bounded_scheduling import timeline_records
from pydantic import Field
from roboflow_workflows.execution_engine.v2 import (
    Block,
    BlockParams,
    Catalogue,
    CompileOptions,
    Output,
    PipelineOptions,
    Ref,
    compile_workflow,
    phase,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND

WAIT_SECONDS = 30
CORRECT_TOTALS = [0.0, 10.0]


class Counter(Block):
    """Ordinary block: state on ``self`` inside one ``run`` is fine."""

    type = "authoring/counter@v1"
    outputs = {"count": Output(FLOAT_KIND, description="Calls so far.")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND) = Field(description="Ignored input.")

    def __init__(self) -> None:
        self.count = 0

    def run(self, *, value: float) -> Dict[str, Any]:
        """Count calls.

        Args:
            value: Ignored.

        Returns:
            ``count``.
        """
        self.count += 1

        return {"count": float(self.count)}


class _TwoPhases(Block):
    """Shared declaration: ``prepare`` then ``finish``, total = value * 10."""

    outputs = {"total": Output(FLOAT_KIND, description="value * 10.")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND) = Field(description="Input value.")

    def __init__(self, *, probe: Probe):
        self.probe = probe

    def run(self, *, value: float) -> Dict[str, Any]:
        """Both phases in order.

        Args:
            value: Input value.

        Returns:
            ``total``.
        """
        result = self.finish(prepare=self.prepare(value=value))

        return result


class ScratchOnSelf(_TwoPhases):
    """INTENTIONALLY UNSAFE: scratch kept on ``self`` between two phases."""

    type = "authoring/scratch_on_self@v1"

    @phase
    def prepare(self, *, value: float) -> float:
        """Store the scratch on the shared instance (the trap)."""
        with self.probe.call(f"prepare:{value:g}"):
            self.scratch = value

        return value

    @phase
    def finish(self, *, prepare: float) -> Dict[str, Any]:
        """Read the scratch back; another pulse may have replaced it."""
        with self.probe.call(f"finish:{prepare:g}"):
            total = self.scratch * 10

        return {"total": total}


class ScratchAsOutput(_TwoPhases):
    """Fix 1: the scratch travels as ``prepare``'s return value."""

    type = "authoring/scratch_as_output@v1"

    @phase
    def prepare(self, *, value: float) -> Dict[str, float]:
        """Return the scratch instead of storing it."""
        with self.probe.call(f"prepare:{value:g}"):
            scratch = {"value": value}

        return scratch

    @phase
    def finish(self, *, prepare: Dict[str, float]) -> Dict[str, Any]:
        """Consume this call's own scratch."""
        with self.probe.call(f"finish:{prepare['value']:g}"):
            total = prepare["value"] * 10

        return {"total": total}


class ScratchOnSelfWholeCall(ScratchOnSelf):
    """Fix 2: the same unsafe phases, gated as one whole call."""

    type = "authoring/scratch_on_self_whole_call@v1"
    phase_overlap = False


AUTHORING_CATALOGUE = Catalogue(
    [Counter, ScratchOnSelf, ScratchAsOutput, ScratchOnSelfWholeCall],
    namespace="authoring",
)


def _definition(block_type: str, output: str) -> Dict[str, Any]:
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "value", "kind": ["float"]}],
        "steps": [{"type": block_type, "name": "block", "value": "$inputs.value"}],
        "outputs": [
            {"type": "JsonField", "name": output, "selector": f"$steps.block.{output}"}
        ],
    }

    return definition


def _session(block_type: str, output: str, *, execution: str, **resources: Any):
    plan = compile_workflow(
        _definition(block_type, output),
        catalogue=AUTHORING_CATALOGUE,
        options=CompileOptions(block_execution=execution),
    )
    session = plan.create_session(resources)

    return session


def _value(result: Any, output: str) -> float:
    (row,) = result.rows()
    value = row[output]

    return value


def ordinary_counter() -> Dict[str, Any]:
    """Four pipelined submissions of a run-mode counter count 1, 2, 3, 4."""
    session = _session(Counter.type, "count", execution="run")
    with session.pipeline(options=PipelineOptions(max_in_flight=3)) as pipeline:
        futures = [pipeline.submit({"value": float(n)}) for n in range(4)]
        counts = [_value(f.result(timeout=WAIT_SECONDS), "count") for f in futures]

    expect("ordinary counter, in submission order", counts, [1.0, 2.0, 3.0, 4.0])
    evidence = {"counts": counts, "expected": [1.0, 2.0, 3.0, 4.0]}

    return evidence


def two_submissions(block_type: str, *, force_overlap: bool) -> Dict[str, Any]:
    """Submit values 0 and 1; hold pulse 0 inside ``finish``.

    Args:
        block_type: One of the two-phase blocks.
        force_overlap: Keep pulse 0 in ``finish`` until pulse 1 has entered
            ``prepare``. Only valid for blocks whose phases may overlap; with
            ``phase_overlap = False`` that can never happen.

    Returns:
        Pipelined and serial totals, the overlap observed and the timeline.
    """
    probe = Probe()
    probe.hold("finish:0")
    session = _session(block_type, "total", execution="phases", probe=probe)
    with session.pipeline(options=PipelineOptions(max_in_flight=2)) as pipeline:
        first = pipeline.submit({"value": 0.0})
        probe.reached("finish:0")
        second = pipeline.submit({"value": 1.0})
        try:
            if force_overlap:
                probe.reached("prepare:1")
        finally:
            probe.release("finish:0")
        totals = [
            _value(f.result(timeout=WAIT_SECONDS), "total") for f in (first, second)
        ]

    serial = _session(block_type, "total", execution="phases", probe=Probe())
    serial_totals = [_value(serial.run({"value": v}), "total") for v in (0.0, 1.0)]
    evidence = {
        "pipelined_totals": totals,
        "serial_totals": serial_totals,
        "correct_totals": CORRECT_TOTALS,
        "pulse_1_prepared_while_pulse_0_in_finish": (
            ("finish:0", "prepare:1") in probe.overlaps()
        ),
        "stages": sorted(pipeline.counters.snapshot()["stages"]),
        "timeline": timeline_records(probe),
    }

    return evidence


def authoring() -> Dict[str, Any]:
    """The trap and both fixes side by side, plus the ordinary counter.

    Returns:
        Evidence per block; each states what it shows.
    """
    trap = two_submissions(ScratchOnSelf.type, force_overlap=True)
    expect("trap: serial run is correct", trap["serial_totals"], CORRECT_TOTALS)
    expect("trap: pipelined result is wrong", trap["pipelined_totals"], [10.0, 10.0])
    trap["shows"] = (
        "INTENTIONALLY UNSAFE block: pulse 1 overwrote self.scratch while pulse 0 "
        "was between phases. Silent wrong value, no error; the serial run is "
        "correct, so only a pipelined phase-mode test reveals it."
    )

    as_output = two_submissions(ScratchAsOutput.type, force_overlap=True)
    expect(
        "fix 1: same overlap",
        as_output["pulse_1_prepared_while_pulse_0_in_finish"],
        True,
    )
    expect("fix 1: correct", as_output["pipelined_totals"], CORRECT_TOTALS)
    as_output["shows"] = (
        "Scratch passed as prepare's return value: the same forced overlap, "
        "correct totals. Phases still overlap across pulses."
    )

    whole_call = two_submissions(ScratchOnSelfWholeCall.type, force_overlap=False)
    expect("fix 2: correct", whole_call["pipelined_totals"], CORRECT_TOTALS)
    expect(
        "fix 2: no phase overlap",
        whole_call["pulse_1_prepared_while_pulse_0_in_finish"],
        False,
    )
    expect("fix 2: one whole-call stage", whole_call["stages"], ["$steps.block#call"])
    whole_call["shows"] = (
        "Same unsafe phases with phase_overlap = False: one stage holds the whole "
        "call, so pulse 1 prepares only after pulse 0 finished."
    )

    evidence = {
        "trap: scratch on self (intentionally unsafe)": trap,
        "fix 1: scratch as phase output": as_output,
        "fix 2: phase_overlap = False": whole_call,
        "ordinary counter (run mode)": ordinary_counter(),
    }

    return evidence


__all__: List[str] = ["authoring", "two_submissions", "ordinary_counter"]
