"""Synthetic, event-driven probe blocks and sources for the scheduling examples.

SYNTHETIC WORKLOAD: these blocks compute nothing useful. They record when
each call enters and leaves, and a call can be held until the example
releases it. Every ordering the examples show is forced by those events;
nothing sleeps and no timing is claimed::

    probe.hold("second:a0")                  # this call will wait inside
    ... start a run ...
    probe.reached("second:a0")               # it is inside now
    probe.reached("first:a1")                # the next frame entered phase first
    probe.release("second:a0")

``Probe.timeline`` is the order of ``enter``/``leave`` events. Two calls
overlapped when one entered before the other left.
"""

import threading
from collections import defaultdict
from contextlib import contextmanager
from fractions import Fraction
from typing import Any, Dict, Iterator, List, Optional, Tuple

from pydantic import Field, StrictInt
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.context import current_pulse_run_id
from roboflow_workflows.execution_engine.v2.data import Timestamp
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from roboflow_workflows.execution_engine.v2.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

WAIT_SECONDS = 30
"""Anti-hang bound of every probe wait; examples finish far earlier."""

CLOCK = "probe-frames"


class ProbeTimeout(AssertionError):
    """A probe event did not happen within ``WAIT_SECONDS``."""


class ObservationMismatch(AssertionError):
    """An example observed something other than what it documents."""


def expect(label: str, actual: Any, expected: Any) -> Any:
    """Raise when an observation differs from the documented value.

    Args:
        label: What is checked, shown on mismatch.
        actual: Observed value.
        expected: Documented value.

    Returns:
        ``actual``.

    Raises:
        ObservationMismatch: When the values differ.
    """
    if actual != expected:
        raise ObservationMismatch(f"{label}: expected {expected!r}, got {actual!r}")

    return actual


class Probe:
    """Shared by one session's probe blocks and sources (a constructor resource).

    Attributes:
        timeline: ``(sequence, event, key, run_id)`` in the order they happened;
            ``event`` is ``enter`` or ``leave``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.timeline: List[Tuple[int, str, str, Optional[str]]] = []
        self._reached: Dict[str, threading.Event] = defaultdict(threading.Event)
        self._held: Dict[str, threading.Event] = {}

    def hold(self, *keys: str) -> None:
        """Make the calls named ``keys`` wait inside until released."""
        for key in keys:
            self._held[key] = threading.Event()

    def release(self, *keys: str) -> None:
        """Let held calls continue."""
        for key in keys:
            self._held[key].set()

    def reached(self, key: str) -> None:
        """Wait until the call ``key`` has entered.

        Raises:
            ProbeTimeout: When it does not enter within ``WAIT_SECONDS``.
        """
        with self._lock:
            event = self._reached[key]
        if not event.wait(WAIT_SECONDS):
            raise ProbeTimeout(f"{key} was not reached within {WAIT_SECONDS} s")

    def entered(self, key: str) -> bool:
        """Whether the call ``key`` has entered (no waiting)."""
        with self._lock:
            entered = self._reached[key].is_set()

        return entered

    @contextmanager
    def call(self, key: str) -> Iterator[None]:
        """Record one call: ``enter``, wait if held, run the body, ``leave``."""
        with self._lock:
            self._record("enter", key)
            reached = self._reached[key]
            held = self._held.get(key)
        reached.set()
        if held is not None and not held.wait(WAIT_SECONDS):
            raise ProbeTimeout(f"{key} was held and never released")
        try:
            yield
        finally:
            with self._lock:
                self._record("leave", key)

    def overlaps(self) -> List[Tuple[str, str]]:
        """Pairs of calls that were inside at the same time, in entry order."""
        inside: List[str] = []
        pairs: List[Tuple[str, str]] = []
        for _, event, key, _ in self.timeline:
            if event == "enter":
                pairs.extend((other, key) for other in inside)
                inside.append(key)
            else:
                inside.remove(key)

        return pairs

    def _record(self, event: str, key: str) -> None:
        self.timeline.append((len(self.timeline), event, key, current_pulse_run_id()))


class ProbeStage(Block):
    """SYNTHETIC: one whole call per invocation; passes ``value`` on."""

    type = "probe/stage@v1"
    outputs = {"value": Output(FLOAT_KIND, description="The input value.")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND) = Field(description="Frame value.")
        tag: Ref(STRING_KIND) = Field(description="Frame tag, e.g. a0.")
        fail_on: str = Field(default="", description="Tag that raises.")

    def __init__(self, *, probe: Probe):
        self.probe = probe
        self.seen: List[str] = []

    def run(self, *, value: float, tag: str, fail_on: str) -> Dict[str, Any]:
        """Record the call; fail for ``fail_on``.

        Args:
            value: Frame value.
            tag: Frame tag.
            fail_on: Tag that raises ``RuntimeError``.

        Returns:
            ``value`` unchanged.
        """
        with self.probe.call(f"{self.execution_context.step_path[-1]}:{tag}"):
            self.seen.append(tag)
            if tag == fail_on:
                raise RuntimeError(f"probe failure requested for {tag}")

        return {"value": value}


class ProbeModel(Block):
    """SYNTHETIC: two phases, ``first`` then ``second``, standing in for a model."""

    type = "probe/two_phase_model@v1"
    outputs = {"score": Output(FLOAT_KIND, description="value * 10.")}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND) = Field(description="Frame value.")
        tag: Ref(STRING_KIND) = Field(description="Frame tag.")

    def __init__(self, *, probe: Probe):
        self.probe = probe

    @phase
    def first(self, *, value: float, tag: str) -> float:
        """Stand-in for pre-processing."""
        with self.probe.call(f"first:{tag}"):
            prepared = value

        return prepared

    @phase
    def second(self, *, first: float, tag: str) -> Dict[str, Any]:
        """Stand-in for the forward pass and post-processing."""
        with self.probe.call(f"second:{tag}"):
            result = {"score": first * 10}

        return result

    def run(self, *, value: float, tag: str) -> Dict[str, Any]:
        """Both phases in order.

        Args:
            value: Frame value.
            tag: Frame tag.

        Returns:
            ``score``.
        """
        result = self.second(first=self.first(value=value, tag=tag), tag=tag)

        return result


class ProbeWholeCallModel(ProbeModel):
    """SYNTHETIC: the same phases, opted out of overlap (one gate per call)."""

    type = "probe/two_phase_model_whole_call@v1"
    phase_overlap = False


class ProbeFrames(Source):
    """SYNTHETIC finite source: ``count`` frames tagged ``<prefix><n>``, 40 ms apart.

    ``read`` of frame n is a probe call ``read:<prefix><n>``, so an example
    can hold the reader or wait until a frame was read.
    """

    type = "probe/frames@v1"
    outputs = {
        "value": SourceOutput(FLOAT_KIND, description="Frame number."),
        "tag": SourceOutput(STRING_KIND, description="Frame tag."),
    }

    class Params(SourceParams):
        count: StrictInt = Field(gt=0, description="Frames to emit.")
        prefix: str = Field(description="Tag prefix, e.g. a.")

    def __init__(self, *, probe: Probe):
        self.probe = probe

    def open(self, *, count: int, prefix: str) -> None:
        """Start at frame 0.

        Args:
            count: Frames to emit.
            prefix: Tag prefix.
        """
        self.count = count
        self.prefix = prefix
        self.position = 0

    def read(self) -> Optional[Emission]:
        """Emit the next frame, or None after the last one or on stop.

        Returns:
            One emission with media PTS ``40 ms * n``.
        """
        if self.stop_event.is_set() or self.position == self.count:
            return None

        number = self.position
        self.position += 1
        tag = f"{self.prefix}{number}"
        with self.probe.call(f"read:{tag}"):
            emission = Emission(
                {"value": float(number), "tag": tag},
                media=Timestamp(
                    ticks=40 * number, time_base=Fraction(1, 1000), clock_id=CLOCK
                ),
            )

        return emission


PROBE_CATALOGUE = Catalogue(
    [ProbeStage, ProbeModel, ProbeWholeCallModel],
    sources=[ProbeFrames],
    namespace="probe",
)
"""Catalogue of the synthetic probe blocks and source."""
