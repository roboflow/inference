"""WP-A05: the usage side of the stream session identity.

``inference.usage_tracking.stream_session`` is the historical name of the
stream session module; the collector and the Workflows observer read the
pipeline's session id through it. They must share the stream's one
ContextVar, and a pipeline's id must reach usage rows recorded in step pools.
"""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import List
from unittest import mock
from unittest.mock import MagicMock

from inference.core.interfaces import workflows_execution_observer
from inference.core.interfaces.legacy_stream import inference_pipeline
from inference.core.interfaces.stream import session
from inference.core.interfaces.stream.pipeline import InferencePipeline
from inference.core.interfaces.workflows_execution_observer import (
    UsageTrackingExecutionObserver,
)
from inference.usage_tracking import collector
from inference.usage_tracking import stream_session as legacy_session
from inference.usage_tracking.collector import usage_collector

PROJECT_ROOT = Path(__file__).resolve().parents[4]

# Module-name fragments the historical usage path must never load.
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


def test_historical_usage_path_re_exports_the_exact_session_objects() -> None:
    assert legacy_session.stream_session_id is session.stream_session_id
    assert legacy_session.mint_stream_session_id is session.mint_stream_session_id


def test_every_host_consumer_binds_the_one_context_variable() -> None:
    assert collector.stream_session_id_var is session.stream_session_id
    assert workflows_execution_observer.stream_session_id is session.stream_session_id
    # Historical monkeypatch points of the legacy pipeline module.
    assert inference_pipeline.stream_session_id is session.stream_session_id
    assert inference_pipeline.mint_stream_session_id is session.mint_stream_session_id


def test_importing_the_historical_usage_path_stays_import_light() -> None:
    assert "inference.usage_tracking.collector" in HEAVY_MODULE_MARKERS
    assert loaded_heavy_modules("inference.usage_tracking.stream_session") == []


def test_step_pool_workers_record_usage_under_the_pipeline_session() -> None:
    # given - the Workflows observer hands the session to a reused pool thread
    # that still holds a previous pipeline's id
    observer = UsageTrackingExecutionObserver()
    recorded = usage_collector.empty_usage_dict(exec_session_id="test-session")
    seen = {}

    with ThreadPoolExecutor(max_workers=1) as step_pool:
        step_pool.submit(session.stream_session_id.set, "stale-stream").result()

        def step(context):
            with observer.step_scope(context=context, step_name="a_step"):
                seen["worker"] = session.stream_session_id.get()
                usage_collector.record_usage(
                    source="a_step",
                    category="workflow_block",
                    api_key="session-key",
                    resource_id="a_step",
                )

        def on_video_frame(video_frames):
            context = observer.capture_step_context()
            step_pool.submit(step, context).result()
            return []

        pipeline = InferencePipeline(
            on_video_frame=on_video_frame,
            video_sources=[],
            predictions_queue=Queue(maxsize=8),
            watchdog=MagicMock(),
            status_update_handlers=[],
        )
        pipeline._generate_frames = lambda: iter([[MagicMock()]])

        # when
        with mock.patch.object(usage_collector, "_usage", recorded):
            thread = Thread(target=pipeline._execute_inference)
            thread.start()
            thread.join(timeout=10)

    # then
    rows = [row for per_key in recorded.values() for row in per_key.values()]
    assert seen["worker"] == pipeline._stream_session_id
    assert [row["stream_session_id"] for row in rows] == [pipeline._stream_session_id]
