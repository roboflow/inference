"""Workflows without handlers keep their behavior and pay for nothing new."""

import subprocess
import sys
import textwrap
import threading
from pathlib import Path

from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.reactions.runtime import session_reactions

from tests.unit_tests.execution_engine.v2.reactions.test_runtime import (
    CATALOGUE,
    PASSIVE,
    WAIT,
    Probe,
    active_definition,
)

WORKFLOWS = Path(__file__).resolve().parents[5]


def test_an_active_run_without_handlers_has_no_reaction_runtime_or_threads():
    probe = Probe()
    plan = compile_workflow(active_definition(), catalogue=CATALOGUE)
    session = plan.create_session(
        resources={"probe": probe, "feeds": {"cam_a": [0, 1]}}
    )
    run = session.start()

    names = {thread.name for thread in threading.enumerate()}
    assert run.wait(WAIT)

    assert run._reactions is None
    assert run.reaction_counters == {} and run.reaction_outcomes() == ()
    assert not [name for name in names if "reaction" in name]
    # The declared event was validated and dropped: nobody listens.
    assert probe.values("emitted") == [0.0, 1.0]


def test_a_passive_session_without_handlers_has_no_reaction_runtime():
    probe = Probe()
    plan = compile_workflow(PASSIVE, catalogue=CATALOGUE)
    session = plan.create_session(resources={"probe": probe})

    assert session.run({"value": 1.0}).rows() == [{"value": 1.0}]
    assert session_reactions(session) is None


def test_a_plain_workflow_imports_neither_state_nor_redis():
    script = textwrap.dedent(
        """
        import sys, threading
        from tests.unit_tests.execution_engine.v2.reactions.test_runtime import (
            CATALOGUE, Probe, active_definition,
        )
        from roboflow_workflows.execution_engine.v2.compilation import compile_workflow

        plan = compile_workflow(active_definition(), catalogue=CATALOGUE)
        session = plan.create_session(resources={"probe": Probe(),"""
        """ "feeds": {"cam_a": [0]}})
        run = session.start()
        assert run.wait(10)
        loaded = [name for name in sys.modules if name == "redis" or"""
        """ ".v2.state" in name]
        print(loaded)
        """
    )
    environment_path = [str(WORKFLOWS.parent), str(WORKFLOWS)]
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=WORKFLOWS,
        env={"PYTHONPATH": ":".join(environment_path), "PYTHONDONTWRITEBYTECODE": "1"},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "[]"
