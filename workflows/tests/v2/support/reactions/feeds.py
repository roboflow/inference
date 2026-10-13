"""Scripted feed source shared by reaction runtime tests."""

import threading
from typing import Dict, Optional

from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine.sources import (
    Emission,
    Source,
    SourceOutput,
    SourceParams,
)

WAIT = 10.0


class Feed(Source):
    """Emits the values of ``feeds[feed]``; an ``Event`` item waits until set."""

    type = "test/feed@v1"
    outputs = {"value": SourceOutput(FLOAT_KIND)}

    class Params(SourceParams):
        feed: str

    def __init__(self, *, feeds: Dict[str, list]) -> None:
        self.feeds = feeds
        self.items = iter(())

    def open(self, *, feed) -> None:
        self.items = iter(self.feeds[feed])

    def read(self) -> Optional[Emission]:
        item = next(self.items, None)
        while isinstance(item, threading.Event):
            assert item.wait(WAIT)
            item = next(self.items, None)
        if item is None:
            return None
        return Emission({"value": float(item)})

    def close(self) -> None:
        pass
