"""Explicit ordering instrumentation for reproducible capacity examples.

Ordinary sources never wait for each other. These subclasses exist only so a
case can force one legal but adverse arrival order: the follower camera reads
its whole schedule before the leader camera reads anything. They order reads;
they do not pair, buffer or combine samples.
"""

from threading import Event

from sources import TensorCamera


class ReadOrder:
    """Event shared by the two probe cameras of one run."""

    def __init__(self):
        self.follower_closed = Event()


class LeaderAfterFollower(TensorCamera):
    """Leader camera that starts reading only after the follower has closed."""

    type = "temporal_demo/leader_after_follower"

    def __init__(self, *, read_order: ReadOrder):
        """Receive the shared ordering event.

        Args:
            read_order: Event set when the follower camera closes.
        """
        self.read_order = read_order

    def read(self):
        """Wait for the follower, then behave as the ordinary camera.

        Returns:
            The ordinary emission, or None on stop/end.
        """
        while not self.read_order.follower_closed.wait(timeout=0.01):
            if self.stop_event.is_set():
                return None

        emission = super().read()

        return emission


class FollowerFirst(TensorCamera):
    """Follower camera that announces when it has read everything and closed."""

    type = "temporal_demo/follower_first"

    def __init__(self, *, read_order: ReadOrder):
        """Receive the shared ordering event.

        Args:
            read_order: Event to set on close.
        """
        self.read_order = read_order

    def close(self) -> None:
        """Release the leader camera."""
        self.read_order.follower_closed.set()
