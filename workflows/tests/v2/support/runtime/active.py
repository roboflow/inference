"""Scripted sources, collectors and builders for active-run tests."""

import threading
from fractions import Fraction
from typing import Any, Callable, Dict, List, Optional

from roboflow_workflows.v2.data.batch import Timestamp
from roboflow_workflows.v2.data.kinds import FLOAT_KIND, STRING_KIND
from roboflow_workflows.v2.engine.active.runtime import GroupResult
from roboflow_workflows.v2.engine.declaration import Ref
from roboflow_workflows.v2.engine.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

WAIT = 10.0


MEDIA = Fraction(1, 1000)


class Log(list):
    """Append-only event log that lets a test wait for a condition on it."""

    def __init__(self) -> None:
        super().__init__()
        self.changed = threading.Condition()

    def append(self, item: Any) -> None:
        with self.changed:
            super().append(item)
            self.changed.notify_all()

    def wait_for(self, predicate: Callable[[], bool]) -> None:
        with self.changed:
            assert self.changed.wait_for(predicate, timeout=WAIT), list(self)

    def count(self, kind: str, feed: str) -> int:
        return sum(1 for item in self if item[:2] == (kind, feed))


UNTIL_STOP = object()


class FeedSource(Source):
    """Reads a scripted feed: ``Emission`` values, ``Event`` gates, ``Exception``s.

    A gate blocks ``read`` until the test sets it (``UNTIL_STOP`` until the
    run is stopping); an exception is raised by ``read``. ``open`` waits for
    the ``started`` resource, which the harness sets once ``start`` returned,
    so a handler never runs before the test holds the run. Every lifecycle
    call is appended to the ``log`` resource.
    """

    class Params(SourceParams):
        feed: str
        scale: float | Ref(FLOAT_KIND) = 1.0
        fail_open: bool = False
        fail_close: bool = False

    def __init__(
        self, *, feeds: Dict[str, list], log: Log, started: threading.Event
    ) -> None:
        self.feeds = feeds
        self.log = log
        self.started = started
        self.feed = ""
        self.fail_close = False
        self.items = iter(())

    def open(self, *, feed, scale, fail_open, fail_close) -> None:
        self.started.wait(WAIT)
        open_gate = self.feeds.get(f"{feed}:open")
        if open_gate is not None:
            open_gate.wait(WAIT)
        self.feed = feed
        self.fail_close = fail_close
        self.log.append(("open", feed, scale, id(self)))
        if fail_open:
            raise RuntimeError(f"cannot open {feed}")
        self.items = iter(self.feeds[feed])

    def read(self) -> Optional[Emission]:
        item = next(self.items, None)
        while isinstance(item, threading.Event) or item is UNTIL_STOP:
            gate = self.stop_event if item is UNTIL_STOP else item
            self.log.append(("wait", self.feed))
            gate.wait(WAIT)
            item = next(self.items, None)
        if isinstance(item, Exception):
            raise item
        self.log.append(
            ("read", self.feed, None if item is None else sorted(item.data))
        )
        return item

    def close(self) -> None:
        self.log.append(("close", self.feed))
        if self.fail_close:
            raise RuntimeError(f"cannot close {self.feed}")


class Scripted(FeedSource):
    """Three ungrouped ports emitted together or separately."""

    type = "test/scripted@v1"
    outputs = {
        "value": SourceOutput(FLOAT_KIND),
        "label": SourceOutput(STRING_KIND),
        "extra": SourceOutput(),
    }


def emit(
    *, pts: Optional[int] = None, meta: Optional[dict] = None, **ports
) -> Emission:
    media = Timestamp(pts, MEDIA, "media") if pts is not None else None
    return Emission(ports, media=media, source_metadata=meta or {})


def source(name: str, *, feed: Optional[str] = None, **params: Any) -> dict:
    return {"type": Scripted.type, "name": name, "feed": feed or name, **params}


def group(name: str, anchor: str, **fields: str) -> dict:
    return {
        "type": "OutputGroup",
        "name": name,
        "anchor": anchor,
        "outputs": [
            {"type": "JsonField", "name": field, "selector": selector}
            for field, selector in fields.items()
        ],
    }


def active(sources: list, steps: list, groups: list, *, inputs: list = ()) -> dict:
    definition = {
        "version": "2.0",
        "inputs": list(inputs),
        "sources": sources,
        "steps": steps,
        "outputs": groups,
    }

    return definition


class Collector:
    """Handlers that keep every result per group, in delivery order."""

    def __init__(self) -> None:
        self.results: Dict[str, List[GroupResult]] = {}

    def handler(self, name: str) -> Callable[[GroupResult], None]:
        def handle(result: GroupResult) -> None:
            self.results.setdefault(name, []).append(result)

        return handle

    def handlers(self, *names: str) -> Dict[str, Callable[[GroupResult], None]]:
        return {name: self.handler(name) for name in names}

    def rows(self, name: str) -> List[dict]:
        return [result.rows()[0] for result in self.results.get(name, [])]

    def sequences(self, name: str) -> List[int]:
        return [result.pulse.sequence for result in self.results.get(name, [])]
