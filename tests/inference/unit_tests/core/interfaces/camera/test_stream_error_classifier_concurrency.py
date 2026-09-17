"""Concurrency and lifetime guarantees for the native (fd 2) stderr capture.

Every test here runs its workers on daemon threads and joins them with a hard
timeout, or isolates them in a subprocess, so a regression that re-introduces an
unbounded block fails the suite instead of hanging it.
"""

import os
import pathlib
import subprocess
import sys
import threading
import time
from typing import Callable, List, Optional

import pytest

from inference.core.interfaces.camera import stream_error_classifier, video_source
from inference.core.interfaces.camera.stream_error_classifier import (
    capture_process_stderr,
)

# Long enough that a slow CI box never trips it, short enough that a real
# deadlock is reported quickly.
HARD_TIMEOUT_SECONDS = 20.0

# A child pays for a fresh interpreter and the `inference` import before it
# reaches the code under test, so it gets a larger budget - still a hard bound,
# so a wedge fails the test rather than hanging the run.
SUBPROCESS_TIMEOUT_SECONDS = 120.0

# Linux pipes hold 64 KiB by default; write comfortably past that.
OVER_PIPE_CAPACITY_LINES = 200
LINE_PAYLOAD = b"x" * 1024 + b"\n"


def _reset_module_state() -> None:
    # Bounded: a wedged capture must not turn cleanup into a second hang.
    reset = getattr(stream_error_classifier, "_reset_capture_state_for_tests", None)
    if reset is None:
        return
    worker = threading.Thread(target=reset, daemon=True)
    worker.start()
    worker.join(timeout=HARD_TIMEOUT_SECONDS)


@pytest.fixture(autouse=True)
def restore_process_stderr():
    """Undo any fd 2 damage a failing test leaves behind."""
    real_stderr_fd = os.dup(2)
    try:
        yield
    finally:
        _reset_module_state()
        os.dup2(real_stderr_fd, 2)
        os.close(real_stderr_fd)


def _wait_for_drains_to_exit() -> None:
    """Block until no drain thread is still holding its descriptors.

    A capture's teardown does not join its drain thread - the thread notices
    the stop itself and closes the read end on its way out - so a descriptor
    count taken the instant a capture returns can still see one on the way out.
    That is a measurement artefact, not a leak, but it has to be waited out
    before counting or the count is a race.
    """

    def drains_alive() -> bool:
        return any(
            thread.name == "native-stderr-capture" and thread.is_alive()
            for thread in threading.enumerate()
        )

    deadline = time.monotonic() + HARD_TIMEOUT_SECONDS
    while drains_alive() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not drains_alive()


def _call_on_daemon_thread(target: Callable[[], None]) -> dict:
    """Run ``target`` where a wedge cannot take the test runner with it."""
    outcome: dict = {}

    def runner() -> None:
        try:
            target()
        except BaseException as error:  # noqa: BLE001 - the point is to catch it
            outcome["error"] = error

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join(timeout=HARD_TIMEOUT_SECONDS)
    assert not thread.is_alive(), "the worker did not finish within the timeout"
    return outcome


def _open_fd_count() -> int:
    try:
        return len(os.listdir("/proc/self/fd"))
    except OSError:  # pragma: no cover - non-Linux
        pytest.skip("/proc/self/fd is unavailable on this platform")


def _stderr_identity() -> tuple:
    stat = os.fstat(2)
    return stat.st_dev, stat.st_ino


def _run_daemon_threads(targets: List[Callable[[], None]]) -> None:
    threads = [threading.Thread(target=target, daemon=True) for target in targets]
    for thread in threads:
        thread.start()
    stuck = []
    for index, thread in enumerate(threads):
        thread.join(timeout=HARD_TIMEOUT_SECONDS)
        if thread.is_alive():
            stuck.append(index)
    assert not stuck, f"worker threads did not finish within the timeout: {stuck}"


def _run_in_subprocess(script: str) -> subprocess.CompletedProcess:
    """Run a snippet against this interpreter's imports, with a hard timeout."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
    return subprocess.run(
        [sys.executable, "-u", "-c", script],
        capture_output=True,
        text=True,
        timeout=SUBPROCESS_TIMEOUT_SECONDS,
        env=environment,
    )


def test_second_capture_entering_before_first_leaves_does_not_block() -> None:
    # Forces the exact interleaving that wedges an unsynchronised
    # dup2-based capture: A enters, B enters, A leaves, B leaves.
    first_entered = threading.Event()
    second_entered = threading.Event()
    first_left = threading.Event()
    second_left = threading.Event()

    def first_worker() -> None:
        with capture_process_stderr():
            first_entered.set()
            second_entered.wait(timeout=HARD_TIMEOUT_SECONDS)
        first_left.set()

    def second_worker() -> None:
        first_entered.wait(timeout=HARD_TIMEOUT_SECONDS)
        with capture_process_stderr():
            second_entered.set()
            first_left.wait(timeout=HARD_TIMEOUT_SECONDS)
        second_left.set()

    _run_daemon_threads([first_worker, second_worker])

    assert first_left.is_set(), "the first capture never returned from its exit path"
    assert second_left.is_set(), "the second capture never returned from its exit path"


def test_capture_does_not_block_when_output_exceeds_pipe_capacity() -> None:
    captured: List[str] = []

    def worker() -> None:
        with capture_process_stderr() as chunks:
            for _ in range(OVER_PIPE_CAPACITY_LINES):
                os.write(2, LINE_PAYLOAD)
        captured.append("".join(chunks))

    _run_daemon_threads([worker])

    assert captured, "the capture body never completed"
    # The buffer is bounded, so do not require every byte back - only that far
    # more than one pipe's worth survived and nothing blocked.
    assert captured[0].count("x") > 128 * 1024


def test_drains_when_the_read_descriptor_is_above_the_select_limit() -> None:
    # select() raises ValueError for descriptors at or above FD_SETSIZE (1024),
    # which a server holding many cameras and sockets reaches; a capture that
    # cannot test its pipe for readability never drains and never restores fd 2.
    # Isolated in a subprocess because the failure mode is an unbounded block.
    script = """
import os, sys
from inference.core.interfaces.camera.stream_error_classifier import (
    capture_process_stderr,
)

hogs = []
while True:
    fd = os.open(os.devnull, os.O_RDONLY)
    hogs.append(fd)
    if fd > 1100:
        break
for fd in hogs[-4:]:
    os.close(fd)

with capture_process_stderr() as chunks:
    os.write(2, b"y" * 65536)
sys.stdout.write("CAPTURED=%d" % len("".join(chunks)))
"""
    result = _run_in_subprocess(script)

    assert result.returncode == 0, result.stderr[-2000:]
    assert "CAPTURED=65536" in result.stdout


def test_stderr_fd_is_restored_after_interleaved_concurrent_captures() -> None:
    before = _stderr_identity()
    worker_count = 8
    barrier = threading.Barrier(worker_count, timeout=HARD_TIMEOUT_SECONDS)
    inner_barrier = threading.Barrier(worker_count, timeout=HARD_TIMEOUT_SECONDS)

    def worker(index: int) -> None:
        barrier.wait()
        with capture_process_stderr():
            os.write(2, f"worker-{index}\n".encode("utf-8"))
            # Guarantee every worker is inside the capture at the same time.
            inner_barrier.wait()

    _run_daemon_threads([lambda index=i: worker(index) for i in range(worker_count)])

    assert _stderr_identity() == before


def test_stderr_fd_is_restored_when_capture_bodies_raise() -> None:
    before = _stderr_identity()
    worker_count = 6
    barrier = threading.Barrier(worker_count, timeout=HARD_TIMEOUT_SECONDS)
    inner_barrier = threading.Barrier(worker_count, timeout=HARD_TIMEOUT_SECONDS)
    raised = []

    def worker(index: int) -> None:
        barrier.wait()
        try:
            with capture_process_stderr():
                os.write(2, f"worker-{index}\n".encode("utf-8"))
                inner_barrier.wait()
                if index % 2 == 0:
                    raise RuntimeError("boom")
        except RuntimeError:
            raised.append(index)

    _run_daemon_threads([lambda index=i: worker(index) for i in range(worker_count)])

    assert sorted(raised) == [0, 2, 4]
    assert _stderr_identity() == before


def test_interrupt_during_entry_bookkeeping_releases_the_reference(
    monkeypatch,
) -> None:
    holder_entered = threading.Event()
    holder_release = threading.Event()

    def holder() -> None:
        with capture_process_stderr():
            holder_entered.set()
            holder_release.wait(timeout=HARD_TIMEOUT_SECONDS)

    holder_thread = threading.Thread(target=holder, daemon=True)
    holder_thread.start()
    try:
        assert holder_entered.wait(timeout=HARD_TIMEOUT_SECONDS)
        capture = stream_error_classifier._active_capture
        assert capture is not None and capture.ref_count == 1

        def interrupted_quiesce(self) -> int:
            raise KeyboardInterrupt("interrupted during entry bookkeeping")

        monkeypatch.setattr(
            stream_error_classifier._SharedStderrCapture,
            "quiesce",
            interrupted_quiesce,
        )

        def entrant() -> None:
            with capture_process_stderr():
                pass

        outcome = _call_on_daemon_thread(entrant)
        monkeypatch.undo()
        assert isinstance(outcome.get("error"), KeyboardInterrupt)

        # The interrupted entrant must not have kept its reference.
        assert capture.ref_count == 1
    finally:
        holder_release.set()
        holder_thread.join(timeout=HARD_TIMEOUT_SECONDS)

    assert not holder_thread.is_alive()
    assert stream_error_classifier._active_capture is None


def test_interrupt_during_exit_bookkeeping_restores_stderr(monkeypatch) -> None:
    before = _stderr_identity()
    original = stream_error_classifier._SharedStderrCapture.quiesce
    state = {"armed": False}

    def flaky_quiesce(self) -> int:
        if state["armed"]:
            state["armed"] = False
            raise KeyboardInterrupt("interrupted during exit bookkeeping")
        return original(self)

    monkeypatch.setattr(
        stream_error_classifier._SharedStderrCapture, "quiesce", flaky_quiesce
    )

    def participant() -> None:
        with capture_process_stderr():
            os.write(2, b"payload\n")
            state["armed"] = True

    outcome = _call_on_daemon_thread(participant)
    monkeypatch.undo()
    assert isinstance(outcome.get("error"), KeyboardInterrupt)

    assert stream_error_classifier._active_capture is None
    assert _stderr_identity() == before


def test_each_participant_sees_the_output_of_its_own_window() -> None:
    worker_count = 4
    barrier = threading.Barrier(worker_count, timeout=HARD_TIMEOUT_SECONDS)
    inner_barrier = threading.Barrier(worker_count, timeout=HARD_TIMEOUT_SECONDS)
    results: List[Optional[str]] = [None] * worker_count

    def worker(index: int) -> None:
        barrier.wait()
        with capture_process_stderr() as chunks:
            inner_barrier.wait()
            os.write(2, f"marker-for-worker-{index}\n".encode("utf-8"))
        results[index] = "".join(chunks)

    _run_daemon_threads([lambda index=i: worker(index) for i in range(worker_count)])

    for index, text in enumerate(results):
        assert text is not None
        assert f"marker-for-worker-{index}" in text


def test_output_written_before_a_capture_starts_is_not_attributed_to_it() -> None:
    noise_written = threading.Event()
    holder_release = threading.Event()
    late_capture: List[str] = []

    def holder() -> None:
        with capture_process_stderr():
            os.write(2, b"noise-from-an-earlier-open\n")
            noise_written.set()
            holder_release.wait(timeout=HARD_TIMEOUT_SECONDS)

    def latecomer() -> None:
        noise_written.wait(timeout=HARD_TIMEOUT_SECONDS)
        with capture_process_stderr() as chunks:
            os.write(2, b"own-output\n")
        late_capture.append("".join(chunks))
        holder_release.set()

    _run_daemon_threads([holder, latecomer])

    assert late_capture
    assert "own-output" in late_capture[0]
    assert "noise-from-an-earlier-open" not in late_capture[0]


def test_nothing_is_written_into_the_capture_pipe() -> None:
    # Windows are delimited by asking the pipe how much it still holds, never
    # by injecting a marker; a marker can be truncated mid-sequence and then
    # surfaces as garbage in a user-facing error message.
    source = pathlib.Path(stream_error_classifier.__file__).read_text()

    assert '"write"' in source, "sanity check: the write end is still tracked"
    assert "os.write(" not in source, "the module must never write to a descriptor"
    assert "\\x00" not in source


def test_capture_is_a_no_op_when_disabled_by_env(monkeypatch) -> None:
    monkeypatch.setattr(
        stream_error_classifier, "DISABLE_NATIVE_STDERR_CAPTURE", True, raising=False
    )
    before = _stderr_identity()
    seen: List[tuple] = []
    results: List[List[str]] = []

    def worker() -> None:
        with capture_process_stderr() as chunks:
            seen.append(_stderr_identity())
            os.write(2, b"")
        results.append(chunks)

    outcome = _call_on_daemon_thread(worker)

    assert "error" not in outcome, outcome
    assert seen == [before], "fd 2 must not be touched while the switch is off"
    assert results == [[]]
    assert _stderr_identity() == before


def test_repeated_sequential_captures_do_not_leak_file_descriptors() -> None:
    counts: List[int] = []

    def worker() -> None:
        for _ in range(5):
            with capture_process_stderr():
                os.write(2, b"warmup\n")
        _wait_for_drains_to_exit()
        counts.append(_open_fd_count())
        for _ in range(50):
            with capture_process_stderr():
                os.write(2, b"line\n")
        _wait_for_drains_to_exit()
        counts.append(_open_fd_count())

    _run_daemon_threads([worker])

    assert len(counts) == 2
    # Must not grow. It may legitimately shrink: an unrelated descriptor left
    # over from an earlier test can be reaped between the two samples.
    assert counts[1] <= counts[0]


def test_capability_marker_is_exposed_on_the_public_callable() -> None:
    # Integrators that serialised native opens around this helper detect a
    # build that no longer needs the workaround through this attribute, so it
    # has to live on the callable itself, under both names it is imported as.
    assert getattr(capture_process_stderr, "handles_concurrent_opens", False) is True
    assert (
        getattr(video_source.capture_process_stderr, "handles_concurrent_opens", False)
        is True
    )
    assert stream_error_classifier.NATIVE_STDERR_CAPTURE_HANDLES_CONCURRENT_OPENS


def test_interrupt_inside_thread_start_leaves_no_orphan_state(monkeypatch) -> None:
    # Thread.start() may already have spawned the drain when the interrupt
    # lands, so startup has to hand fd 2 back and drop its descriptors without
    # closing a read end the drain now owns.
    before_identity = _stderr_identity()
    before_descriptors = _open_fd_count()
    original_start = threading.Thread.start

    def interrupted_start(self) -> None:
        if self.name == "native-stderr-capture":
            raise KeyboardInterrupt("interrupted inside Thread.start")
        original_start(self)

    monkeypatch.setattr(threading.Thread, "start", interrupted_start)

    def entrant() -> None:
        with capture_process_stderr():
            pass

    outcome = _call_on_daemon_thread(entrant)
    monkeypatch.undo()

    assert isinstance(outcome.get("error"), KeyboardInterrupt)
    assert stream_error_classifier._active_capture is None
    assert _stderr_identity() == before_identity
    _wait_for_drains_to_exit()
    assert _open_fd_count() <= before_descriptors


def test_closing_a_descriptor_gives_up_ownership_of_its_number() -> None:
    # A closed descriptor number is immediately reusable, so a second cleanup
    # path that still remembers it would close something else's file.
    captured: List[stream_error_classifier._SharedStderrCapture] = []

    def worker() -> None:
        with capture_process_stderr():
            capture = stream_error_classifier._active_capture
            assert capture is not None
            captured.append(capture)

    _run_daemon_threads([worker])
    capture = captured[0]
    _wait_for_drains_to_exit()
    assert not capture._fds, "teardown should have given up every descriptor"

    # Take the freed numbers for something unrelated, then run every remaining
    # cleanup path over the spent capture.
    replacements = [os.open(os.devnull, os.O_RDONLY) for _ in range(4)]
    try:
        capture.restore_stderr()
        capture.shut_down()
        capture.discard_in_forked_child()
        for file_descriptor in replacements:
            os.fstat(file_descriptor)  # raises if the capture closed it again
    finally:
        for file_descriptor in replacements:
            os.close(file_descriptor)


def test_capture_is_a_no_op_when_stderr_is_already_closed() -> None:
    # Duplicating the real stderr before allocating the pipe is what stops
    # os.pipe() being handed descriptor 2 and later "restored" onto it.
    #
    # The assertion is "fd 2 is not left as a pipe" rather than "fd 2 is still
    # closed": a background thread of the imported package occasionally opens a
    # socket, which lands on the just-freed descriptor 2 and makes the stricter
    # form flaky (measured about once in sixty runs). Capturing onto whatever
    # genuinely occupies fd 2 is correct behaviour; leaving a pipe there is the
    # defect, and that is what this pins down. (The older single-read
    # implementation ended up closing descriptor 2 rather than leaving a pipe
    # on it, so this is a regression test against the drained design.)
    script = """
import os, stat, sys
from inference.core.interfaces.camera.stream_error_classifier import (
    capture_process_stderr,
)

os.close(2)
with capture_process_stderr() as chunks:
    pass
try:
    mode = os.fstat(2).st_mode
except OSError:
    sys.stdout.write("FD2_CLOSED chunks=%r" % (chunks,))
else:
    kind = "FIFO" if stat.S_ISFIFO(mode) else "OTHER"
    sys.stdout.write("FD2_OPEN_%s chunks=%r" % (kind, chunks))
"""
    result = _run_in_subprocess(script)

    assert result.returncode == 0, result.stderr[-2000:]
    assert "FD2_OPEN_FIFO" not in result.stdout, result.stdout
    assert "chunks=[]" in result.stdout


def test_forked_child_leaves_an_inherited_capture_without_deadlocking() -> None:
    # fork() while another thread holds the capture's condition: the child
    # inherits the locked condition but not the thread that would release it,
    # so unwinding the context there must touch none of the capture's locks.
    script = """
import os, sys, threading, time
from inference.core.interfaces.camera import stream_error_classifier
from inference.core.interfaces.camera.stream_error_classifier import (
    capture_process_stderr,
)

holding = threading.Event()
release = threading.Event()

def hold(capture):
    with capture._condition:
        holding.set()
        release.wait(20)

child = False
with capture_process_stderr():
    capture = stream_error_classifier._active_capture
    threading.Thread(target=hold, args=(capture,), daemon=True).start()
    holding.wait(10)
    pid = os.fork()
    child = pid == 0
    if not child:
        # The child's copy of the condition stays locked forever; let the
        # parent's holder go so the parent can unwind normally.
        release.set()
# both processes unwind the context here
if child:
    os._exit(0)

deadline = time.monotonic() + 10
status = None
while time.monotonic() < deadline:
    finished, status = os.waitpid(pid, os.WNOHANG)
    if finished:
        break
    time.sleep(0.01)
if status is None or not finished:
    os.kill(pid, 9)
    sys.stdout.write("CHILD_STUCK")
else:
    sys.stdout.write("CHILD_EXIT=%d" % status)
"""
    result = _run_in_subprocess(script)

    assert result.returncode == 0, result.stderr[-2000:]
    assert "CHILD_EXIT=0" in result.stdout


def test_fork_closes_the_read_end_of_a_capture_that_is_still_winding_down() -> None:
    # A released capture has already given fd 2 back, but its drain thread is
    # still inside its poll interval and still owns the read end. That thread
    # does not exist in a forked child, so the child must close the descriptor
    # rather than carry it for the rest of its life.
    script = """
import os, stat, sys, time
from inference.core.interfaces.camera.stream_error_classifier import (
    capture_process_stderr,
)

def pipe_fds():
    found = set()
    for name in os.listdir("/proc/self/fd"):
        try:
            file_descriptor = int(name)
            if stat.S_ISFIFO(os.fstat(file_descriptor).st_mode):
                found.add(file_descriptor)
        except (OSError, ValueError):
            pass
    return found

before = pipe_fds()
with capture_process_stderr():
    # A duplicate of the redirected fd 2 is a write end, so the pipe cannot
    # reach EOF and the drain is still polling when the fork happens.
    held_writer = os.dup(2)

pid = os.fork()
if pid == 0:
    os.close(held_writer)
    time.sleep(0.4)
    sys.stdout.write("CHILD_EXTRA_PIPE_FDS=%d " % len(pipe_fds() - before))
    sys.stdout.flush()
    os._exit(0)

os.close(held_writer)
_, status = os.waitpid(pid, 0)
sys.stdout.write("PARENT_STATUS=%d" % status)
"""
    result = _run_in_subprocess(script)

    assert result.returncode == 0, result.stderr[-2000:]
    assert "CHILD_EXTRA_PIPE_FDS=0" in result.stdout, result.stdout
    assert "PARENT_STATUS=0" in result.stdout, result.stdout
