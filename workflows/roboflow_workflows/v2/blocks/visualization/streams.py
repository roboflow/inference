"""A painter stream with an explicit dependency and a ready-on-return boundary.

Image transforms may leave CUDA work queued on the caller's current stream.
Before painting, the side stream waits for that already-queued work. This is
a GPU dependency, not a host wait or a device-wide synchronization. Producers
using other streams must make their outputs ready or join the caller's stream.

The painter queues its whole batch on the side stream and waits once on exit::

    with scope.work(device):
        copy, boxes, labels ...
    # An event behind the queued work has completed; the outputs are ready.

The final wait also runs when painting raises, so queued kernels do not outlive
the call. Canvases, tables and input frames can then be released safely.
"""

import threading
from contextlib import contextmanager
from typing import Dict, Iterator

import torch


class ReadyScope:
    """Run CUDA work on a per-thread side stream; wait for it on scope exit.

    On a non-CUDA device the body runs as is.
    """

    def __init__(self) -> None:
        self._local = threading.local()

    @contextmanager
    def work(self, device: torch.device) -> Iterator[None]:
        """Run the body on this thread's stream of ``device``, then wait for it.

        Args:
            device: Device of the work.

        Yields:
            Nothing; CUDA work queued in the body runs on the side stream.
        """
        if device.type != "cuda":
            yield
            return

        stream = self._stream(device)
        stream.wait_stream(torch.cuda.current_stream(device))
        try:
            with torch.cuda.stream(stream):
                yield
        finally:
            done = torch.cuda.Event()
            done.record(stream)
            done.synchronize()

    def _stream(self, device: torch.device) -> torch.cuda.Stream:
        streams: Dict[torch.device, torch.cuda.Stream] = (
            self._local.__dict__.setdefault("streams", {})
        )
        if device not in streams:
            streams[device] = torch.cuda.Stream(device=device)

        return streams[device]
