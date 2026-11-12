import threading
from typing import Callable, List, Optional

import pytest


class BlockingWaits:
    """Records the waits that would block on a watched object's events.

    A watched ``threading.Event`` answers ``wait()`` at once with its current
    state. When the event is unset and the timeout is not zero, the timeout is
    recorded, so a test can check that a block never waited without spending
    the wait.
    """

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._monkeypatch = monkeypatch
        self.timeouts: List[Optional[float]] = []

    def watch(self, owner: object) -> None:
        """Watch every ``threading.Event`` attribute of an existing object.

        Args:
            owner: Object whose event attributes are watched, such as a block's
                connection state.
        """
        for value in vars(owner).values():
            if isinstance(value, threading.Event):
                self._monkeypatch.setattr(value, "wait", self._recording_wait(value))

    def watch_new(self, owner_class: type) -> None:
        """Watch the events of every instance of a class created from now on.

        Args:
            owner_class: Class whose instances are watched once constructed, for
                state a block creates during a run.
        """
        original_init = owner_class.__init__

        def init(owner, *args, **kwargs) -> None:
            original_init(owner, *args, **kwargs)
            self.watch(owner)

        self._monkeypatch.setattr(owner_class, "__init__", init)

    def _recording_wait(
        self, event: threading.Event
    ) -> Callable[[Optional[float]], bool]:
        def wait(timeout: Optional[float] = None) -> bool:
            is_set = event.is_set()
            if not is_set and (timeout is None or timeout > 0):
                self.timeouts.append(timeout)
            return is_set

        return wait


@pytest.fixture
def blocking_waits(monkeypatch: pytest.MonkeyPatch) -> BlockingWaits:
    return BlockingWaits(monkeypatch)
