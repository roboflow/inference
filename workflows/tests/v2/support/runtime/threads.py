"""Thread probes shared by worker, passive-pipeline and threading suites."""

import threading
from typing import List


def refuse_second_worker(monkeypatch, prefix: str) -> List[threading.Thread]:
    """Make ``Thread.start`` fail for worker 1 of pools named ``prefix``."""
    real_start = threading.Thread.start
    started: List[threading.Thread] = []

    def start(thread: threading.Thread) -> None:
        if thread.name.startswith(prefix) and thread.name.endswith("-1"):
            raise RuntimeError("injected second-worker launch failure")
        real_start(thread)
        if thread.name.startswith(prefix):
            started.append(thread)

    monkeypatch.setattr(threading.Thread, "start", start)

    return started


def live_threads(prefix: str) -> List[str]:
    names = [
        thread.name
        for thread in threading.enumerate()
        if thread.name.startswith(prefix)
    ]

    return names
