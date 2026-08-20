"""Race-safe terminal bookkeeping for video worker jobs."""

import threading
from typing import Callable, Optional


def finish_run_once(
    worker,
    run,
    *,
    outcome: Optional[str],
    stop_timeout_s: float,
    on_stop_failure: Callable[[object, str], None],
) -> bool:
    """Schedule stop and release for one run exactly once.

    Cancellation must not block the poll loop: that loop is also the heartbeat
    channel for sibling jobs. Teardown therefore runs on a coordinator daemon,
    outside ``runs_lock``. A second bounded daemon wraps ``run.stop()`` because
    InferencePipeline.join() itself has no timeout. The caller's failure hook
    must contain the whole process; it is unsafe to drop bookkeeping while the
    old threaded pipeline may still execute.
    """

    with worker.runs_lock:
        if getattr(run, "_finish_started", False):
            return False
        run._finish_started = True

    if outcome is not None:
        run._record_outcome(outcome)
    # Publish terminal intent before the asynchronous terminate() can unblock
    # the pipeline-end watcher. Both JobRun implementations expose this flag.
    run.cancelling = True

    def finalize_run():
        stop_error = []

        def stop_run():
            try:
                run.stop()
            except BaseException as error:  # surfaced to the supervisor below
                stop_error.append(error)

        stop_thread = threading.Thread(
            target=stop_run,
            name=f"video-job-stop-{run.job_id}",
            daemon=True,
        )
        stop_thread.start()
        stop_thread.join(timeout=max(0.0, stop_timeout_s))
        if stop_thread.is_alive():
            on_stop_failure(run, "timeout")
            return

        if stop_error:
            on_stop_failure(run, f"exception: {type(stop_error[0]).__name__}")
            return

        # A delayed terminal callback from an older attempt must not release the
        # execution domain, plaintext claim proof, or retirement decision
        # belonging to a newer same-ID claim already installed in the map.
        worker.release_run(run)

    threading.Thread(
        target=finalize_run,
        name=f"video-job-finalize-{run.job_id}",
        daemon=True,
    ).start()
    return True
