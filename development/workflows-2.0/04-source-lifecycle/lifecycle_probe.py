"""Explicit test instrumentation; ordinary source plugins have no cross-source waits."""

from dataclasses import dataclass, field
from threading import Event

from source_plugins import CsvTemperature, TensorFrames


@dataclass
class DemoControl:
    """Coordinate the lifecycle probe with events rather than scheduling sleeps.

    Attributes:
        begin: Host releases readers after obtaining its ActiveRun handle.
        first_frame: First image callback releases the temperature reader.
        temperature_opened: Ensure both probe sources acquire before stop/failure.
        temperature_closed: Temperature EOF releases the last two images.
        third_image_read: Reader has advanced past the second admitted image.
        lifecycle: Ordered observations from source threads and callbacks.
    """

    begin: Event = field(default_factory=Event)
    first_frame: Event = field(default_factory=Event)
    temperature_opened: Event = field(default_factory=Event)
    temperature_closed: Event = field(default_factory=Event)
    third_image_read: Event = field(default_factory=Event)
    lifecycle: list = field(default_factory=list)


def _wait_ready(event: Event, *, stop_event: Event) -> bool:
    while not event.wait(timeout=0.01):
        if stop_event.is_set():
            return False

    return not stop_event.is_set()


class ProbedTemperature(CsvTemperature):
    """Instrument the CSV source only for explicit lifecycle/stop/failure probes."""

    type = CsvTemperature.type

    def __init__(self, *, control: DemoControl):
        """Receive synchronization instrumentation.

        Args:
            control: Shared probe events and lifecycle observations.
        """
        self.control = control

    def open(self, *, path: str) -> None:
        """Observe acquisition, then delegate to the ordinary source.

        Args:
            path: Fixture CSV path.
        """
        self.control.lifecycle.append("temperature:open")
        super().open(path=path)
        self.control.temperature_opened.set()

    def read(self):
        """Hold the reader until another source produces its first callback.

        Returns:
            The ordinary source emission after release, or None on stop/EOF.
        """
        if not _wait_ready(self.control.begin, stop_event=self.stop_event):
            return None
        if not _wait_ready(self.control.first_frame, stop_event=self.stop_event):
            return None

        emission = super().read()
        if emission is None and not self.stop_event.is_set():
            self.control.lifecycle.append("temperature:eof")

        return emission

    def close(self) -> None:
        """Observe cleanup and release the remaining image reads."""
        super().close()
        self.control.lifecycle.append("temperature:close")
        self.control.temperature_closed.set()


class ProbedFrames(TensorFrames):
    """Hold the final images until the temperature source has closed."""

    type = TensorFrames.type

    def __init__(self, *, control: DemoControl):
        """Receive synchronization instrumentation.

        Args:
            control: Shared probe events and lifecycle observations.
        """
        self.control = control

    def open(self) -> None:
        """Observe acquisition, then initialize the ordinary image source."""
        self.control.lifecycle.append("camera:open")
        super().open()

    def read(self):
        """Release the first frames immediately and final frames after CSV EOF.

        Returns:
            The ordinary image emission, or None on stop/EOF.
        """
        if not _wait_ready(self.control.begin, stop_event=self.stop_event):
            return None
        if self.index == 3 and not _wait_ready(
            self.control.temperature_closed, stop_event=self.stop_event
        ):
            return None

        if self.index == 2:
            self.control.third_image_read.set()
        emission = super().read()
        if emission is None and not self.stop_event.is_set():
            self.control.lifecycle.append("camera:eof")

        return emission

    def close(self) -> None:
        """Record the engine's exactly-once cleanup call."""
        self.control.lifecycle.append("camera:close")
