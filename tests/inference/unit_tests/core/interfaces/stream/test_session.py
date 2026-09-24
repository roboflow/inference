"""WP-A05: the pipeline's stream session identity.

The pipeline binds its session id in its inference thread, where usage
recording reads it. Only stream modules are imported here, so these checks
move with the package; the host-side consumers (the historical usage path, the
collector and the Workflows observer) are covered by
tests/inference/unit_tests/usage_tracking/test_stream_session.py.
"""

import os
import subprocess
import sys
from pathlib import Path
from queue import Queue
from threading import Barrier, Thread
from typing import List, Optional
from unittest import mock
from unittest.mock import MagicMock

import pytest

from inference.core.interfaces.stream import pipeline as pipeline_module
from inference.core.interfaces.stream import session
from inference.core.interfaces.stream.pipeline import InferencePipeline

PROJECT_ROOT = Path(__file__).resolve().parents[6]


def _make_pipeline(
    on_video_frame, exec_session_id: Optional[str] = None
) -> InferencePipeline:
    return InferencePipeline(
        on_video_frame=on_video_frame,
        video_sources=[],
        predictions_queue=Queue(maxsize=8),
        watchdog=MagicMock(),
        status_update_handlers=[],
        exec_session_id=exec_session_id,
    )


def _feed_one_frame(pipeline: InferencePipeline) -> None:
    pipeline._generate_frames = lambda: iter([[MagicMock()]])


def test_pipeline_binds_the_session_module_objects() -> None:
    assert pipeline_module.stream_session_id is session.stream_session_id
    assert pipeline_module.mint_stream_session_id is session.mint_stream_session_id


def test_minted_session_ids_are_distinct() -> None:
    minted = {session.mint_stream_session_id() for _ in range(100)}

    assert len(minted) == 100


def test_concurrent_pipelines_keep_distinct_session_ids() -> None:
    # given - both inference threads are inside a frame at the same time
    both_running = Barrier(2, timeout=10)
    seen = {}

    def on_video_frame_for(name):
        def on_video_frame(video_frames):
            both_running.wait()
            seen[name] = session.stream_session_id.get()
            return []

        return on_video_frame

    pipelines = {name: _make_pipeline(on_video_frame_for(name)) for name in ("a", "b")}
    for pipeline in pipelines.values():
        _feed_one_frame(pipeline)

    # when
    threads = [Thread(target=p._execute_inference) for p in pipelines.values()]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)

    # then
    assert pipelines["a"]._stream_session_id != pipelines["b"]._stream_session_id
    assert seen == {name: p._stream_session_id for name, p in pipelines.items()}


def test_explicit_exec_session_id_wins_over_a_minted_one() -> None:
    seen = {}

    def on_video_frame(video_frames):
        seen["value"] = session.stream_session_id.get()
        return []

    with mock.patch.object(
        pipeline_module,
        "mint_stream_session_id",
        side_effect=AssertionError("minted"),
    ):
        pipeline = _make_pipeline(on_video_frame, exec_session_id="camera-7")
    _feed_one_frame(pipeline)

    pipeline._execute_inference()

    assert pipeline._stream_session_id == "camera-7"
    assert seen["value"] == "camera-7"


@pytest.mark.parametrize("fails", [False, True])
def test_inference_thread_resets_the_session_when_it_ends(fails: bool) -> None:
    # given - a caller thread with its own session bound
    def on_video_frame(video_frames):
        assert session.stream_session_id.get() == pipeline._stream_session_id
        if fails:
            raise RuntimeError("inference failed")
        return []

    pipeline = _make_pipeline(on_video_frame)
    _feed_one_frame(pipeline)
    token = session.stream_session_id.set("outer")
    try:
        # when
        pipeline._execute_inference()

        # then
        assert session.stream_session_id.get() == "outer"
    finally:
        session.stream_session_id.reset(token)


# Module-name fragments a session import must never load: video, the stream
# runtime, the Workflows engine, the usage collector and the HTTP server.
HEAVY_MODULE_MARKERS = (
    "inference.core.interfaces.camera",
    "inference.core.interfaces.stream.pipeline",
    "inference.core.interfaces.http",
    "inference.usage_tracking.collector",
    "execution_engine",
    "fastapi",
)


def loaded_heavy_modules(module: str) -> List[str]:
    """Import ``module`` in a fresh interpreter; return the heavy modules loaded."""
    probe = (
        "import importlib, sys\n"
        f"importlib.import_module({module!r})\n"
        f"markers = {HEAVY_MODULE_MARKERS!r}\n"
        "print('\\n'.join(sorted(m for m in sys.modules "
        "if any(marker in m for marker in markers))))\n"
    )
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "DISABLE_VERSION_CHECK": "True",
        "PYTHONPATH": os.pathsep.join(
            [str(PROJECT_ROOT / "workflows"), str(PROJECT_ROOT / "inference_models")]
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    return result.stdout.split()


def test_importing_the_session_stays_import_light() -> None:
    assert loaded_heavy_modules("inference.core.interfaces.stream.session") == []


def test_import_light_probe_detects_a_heavy_import() -> None:
    # Positive control: the probe would notice the session growing an import.
    assert "inference.core.interfaces.stream.pipeline" in loaded_heavy_modules(
        "inference.core.interfaces.stream.pipeline"
    )
