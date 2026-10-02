"""Author and caller contract of pipelining: options, phase_overlap, session use.

Covers what lane A owns: option validation, the ``phase_overlap`` declaration
and its introspection, passive session exclusion, pulse attribution and the
lazy module boundary. Drivers (active runtime, passive pipeline) test their
own behaviour.
"""

import ast
import subprocess
import sys
import threading
from pathlib import Path
from typing import List

import pytest
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.context import (
    current_pulse_run_id,
    use_pulse_run_id,
)
from roboflow_workflows.execution_engine.v2.declaration import (
    Block,
    BlockParams,
    Output,
    Ref,
)
from roboflow_workflows.execution_engine.v2.errors import (
    ContractError,
    DeclarationError,
    StepExecutionError,
    WorkflowInputError,
)
from roboflow_workflows.execution_engine.v2.execution.steps import RunState
from roboflow_workflows.execution_engine.v2.implementations import Implementation
from roboflow_workflows.execution_engine.v2.kinds import FLOAT_KIND
from roboflow_workflows.execution_engine.v2.phases import phase
from roboflow_workflows.execution_engine.v2.pipelining import PipelineOptions
from roboflow_workflows.execution_engine.v2.pipelining.stages import (
    SERIAL,
    PipelinedCoordination,
    step_stage_units,
)
from roboflow_workflows.execution_engine.v2.plan import CompileOptions

from tests.unit_tests.execution_engine.v2.test_active_runtime import (
    Harness,
    doubling,
    emit,
    group,
    source,
)

TIMEOUT = 5.0


# Options ---------------------------------------------------------------------


def test_default_options_and_per_source_policy() -> None:
    options = PipelineOptions(overload="latest", source_overload={"file": "block"})

    assert PipelineOptions().max_in_flight == 4
    assert PipelineOptions().overload == "block"
    assert options.overload_for("file") == "block"
    assert options.overload_for("camera") == "latest"
    options.check_sources(["file", "camera"])
    with pytest.raises(ContractError, match=r"unknown source\(s\) \['file'\]"):
        options.check_sources(["camera"])
    with pytest.raises(TypeError):
        options.source_overload["camera"] = "block"


@pytest.mark.parametrize(
    "arguments, message",
    [
        ({"max_in_flight": 0}, "at least 1"),
        ({"max_in_flight": True}, "must be an int"),
        ({"max_in_flight": 2.0}, "must be an int"),
        ({"overload": "drop_oldest"}, "overload must be one of"),
        ({"source_overload": [("a", "block")]}, "must map source names"),
        ({"source_overload": {"": "block"}}, "keys are source names"),
        ({"source_overload": {"a": "newest"}}, r"source_overload\['a'\]"),
    ],
)
def test_invalid_options_are_rejected(arguments, message) -> None:
    with pytest.raises(ContractError, match=message):
        PipelineOptions(**arguments)


# phase_overlap -----------------------------------------------------------------


class Shared(Block):
    """Phases sharing scratch state on ``self``: must not overlap."""

    type = "test/pipelining/shared@v1"
    outputs = {"total": Output(FLOAT_KIND)}
    phase_overlap = False

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    @phase
    def first(self, *, value):
        self.scratch = value
        return value

    @phase
    def second(self, *, first):
        return {"total": first + self.scratch}

    def run(self, *, value):
        return self.second(first=self.first(value=value))


class Independent(Implementation):
    name = "independent"

    @phase
    def first(self, *, value):
        return value

    @phase
    def second(self, *, first):
        return {"total": first}

    def run(self, *, value):
        return self.second(first=self.first(value=value))


class Exclusive(Independent):
    name = "exclusive"
    phase_overlap = False


class Contract(Block):
    type = "test/pipelining/contract@v1"
    outputs = {"total": Output(FLOAT_KIND)}
    implementations = (Independent, Exclusive)

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)


def test_phase_overlap_is_declared_copied_and_introspected() -> None:
    shared = Shared.__block_spec__.implementations[0]
    independent, exclusive = Contract.__block_spec__.implementations

    assert Block.phase_overlap is True and Implementation.phase_overlap is True
    assert (shared.phase_overlap, independent.phase_overlap) == (False, True)
    assert exclusive.phase_overlap is False
    described = Contract.__block_spec__.describe()["implementations"]
    assert [item["phase_overlap"] for item in described] == [True, False]


def test_step_stage_units_follow_execution_mode_and_phase_overlap() -> None:
    def units(block: type, mode: str):
        definition = {
            "version": "2.0",
            "inputs": [{"type": "WorkflowParameter", "name": "v", "kind": ["float"]}],
            "steps": [{"type": block.type, "name": "s", "value": "$inputs.v"}],
            "outputs": [],
        }
        plan = compile_workflow(
            definition,
            catalogue=Catalogue([block]),
            options=CompileOptions(block_execution=mode),
        )
        return step_stage_units(plan.steps[0])

    assert units(Contract, "phases") == ("first", "second")
    assert units(Contract, "run") == ("call",)
    assert units(Shared, "phases") == ("call",)


def test_invalid_phase_overlap_declarations_are_rejected() -> None:
    with pytest.raises(DeclarationError, match="phase_overlap must be True or False"):
        type(
            "Bad",
            (Shared,),
            {"type": "test/pipelining/bad@v1", "phase_overlap": "no"},
        )
    with pytest.raises(DeclarationError, match="phase_overlap must be True or False"):
        type(
            "BadContract",
            (Block,),
            {
                "type": "test/pipelining/bad_contract@v1",
                "outputs": {"total": Output(FLOAT_KIND)},
                "Params": Contract.Params,
                "implementations": (
                    type(
                        "BadImpl", (Independent,), {"name": "bad", "phase_overlap": 1}
                    ),
                ),
            },
        )
    with pytest.raises(DeclarationError, match="belongs to each Implementation"):
        type(
            "ContractOverlap",
            (Contract,),
            {"type": "test/pipelining/contract_overlap@v1", "phase_overlap": False},
        )
    with pytest.raises(DeclarationError, match="shadow"):
        type(
            "Shadowing",
            (Implementation,),
            {"name": "x", "phase_overlap": phase(lambda self, *, value: value)},
        )


# Session use -----------------------------------------------------------------


class Waiting(Block):
    """Blocks inside ``run`` until the test releases it."""

    type = "test/pipelining/waiting@v1"
    outputs = {"total": Output(FLOAT_KIND)}

    class Params(BlockParams):
        value: Ref(FLOAT_KIND)

    def __init__(self, *, entered: threading.Event, release: threading.Event):
        self.entered = entered
        self.release = release

    def run(self, *, value):
        self.entered.set()
        self.release.wait(TIMEOUT)
        if value < 0:
            raise ValueError("negative")
        return {"total": value}


def waiting_session():
    definition = {
        "version": "2.0",
        "inputs": [{"type": "WorkflowParameter", "name": "v", "kind": ["float"]}],
        "steps": [{"type": Waiting.type, "name": "s", "value": "$inputs.v"}],
        "outputs": [{"type": "JsonField", "name": "t", "selector": "$steps.s.total"}],
    }
    plan = compile_workflow(definition, catalogue=Catalogue([Waiting]))
    entered, release = threading.Event(), threading.Event()
    session = plan.create_session(resources={"entered": entered, "release": release})

    return session, entered, release


def test_open_pipeline_excludes_direct_runs_and_a_second_pipeline() -> None:
    session, _, release = waiting_session()
    release.set()

    session._claim_pipeline()
    with pytest.raises(ContractError, match="open pipeline; submit to it"):
        session.run({"v": 1.0})
    with pytest.raises(ContractError, match="already has an open pipeline"):
        session._claim_pipeline()

    session._release_pipeline()
    assert session.run({"v": 1.0}).outputs is not None


def test_pipeline_does_not_open_over_a_direct_run_in_progress() -> None:
    session, entered, release = waiting_session()
    errors: List[BaseException] = []

    def direct() -> None:
        try:
            session.run({"v": 2.0})
        except BaseException as error:
            errors.append(error)

    runner = threading.Thread(target=direct, daemon=True)
    runner.start()
    assert entered.wait(TIMEOUT)

    with pytest.raises(ContractError, match="running 1 direct run"):
        session._claim_pipeline()
    release.set()
    runner.join(TIMEOUT)

    assert errors == []
    session._claim_pipeline()
    session._release_pipeline()


def test_failed_direct_run_releases_its_claim() -> None:
    session, _, release = waiting_session()
    release.set()

    with pytest.raises(StepExecutionError):
        session.run({"v": -1.0})

    session._claim_pipeline()
    session._release_pipeline()


def test_pipeline_options_are_type_checked_and_routed_by_plan_kind() -> None:
    passive, _, _ = waiting_session()
    harness = Harness(
        doubling([source("a")], [group("A", "$sources.a.value")]),
        {"a": [emit(value=1.0)]},
    )

    with pytest.raises(ContractError, match="must be PipelineOptions"):
        passive.pipeline(options={"max_in_flight": 2})
    with pytest.raises(WorkflowInputError, match="pipeline=PipelineOptions"):
        harness.session.pipeline()
    with pytest.raises(ContractError, match="must be PipelineOptions"):
        harness.session.start(pipeline=2)
    assert list(harness.log) == []


# Attribution and callbacks -----------------------------------------------------


def test_pulse_run_id_is_scoped_to_the_executing_thread() -> None:
    seen: List[object] = []

    assert current_pulse_run_id() is None
    with use_pulse_run_id("run:a:0"):
        other = threading.Thread(target=lambda: seen.append(current_pulse_run_id()))
        other.start()
        other.join(TIMEOUT)
        with use_pulse_run_id("run:op:0"):
            seen.append(current_pulse_run_id())
        seen.append(current_pulse_run_id())
    with pytest.raises(ValueError):
        with use_pulse_run_id("run:a:1"):
            raise ValueError("pulse failed")

    assert seen == [None, "run:op:0", "run:a:0"]
    assert current_pulse_run_id() is None


def test_run_state_routes_callbacks_through_its_coordination() -> None:
    handled: List[object] = []
    session, _, _ = waiting_session()
    session.error_handler = handled.append
    serial = RunState(session=session, run_id="r", inputs={})
    pipelined = RunState(
        session=session,
        run_id="r",
        inputs={},
        coordination=PipelinedCoordination(session, options=PipelineOptions()),
    )

    assert serial.coordination is SERIAL and serial.ticket is None
    assert serial.observer is session.observer
    assert serial.error_handler is session.error_handler
    # The built-in no-op observer has nothing to serialize (SF-1); a user
    # error handler is still wrapped with the run's callback lock.
    assert pipelined.observer is session.observer
    assert pipelined.error_handler is not session.error_handler
    pipelined.error_handler("failure")
    pipelined.observer.on_run_started(session_id="s", run_id="r")
    assert handled == ["failure"]
    with pytest.raises(ContractError, match="without a ticket"):
        pipelined.coordination.stages(pipelined, {"call": "g"}, calls=1)


def test_scheduling_modules_need_no_optional_dependency() -> None:
    """Core pipelining imports only the standard library and the engine."""
    package = Path(PipelineOptions.__module__.replace(".", "/")).parent
    root = Path(sys.modules["roboflow_workflows"].__file__).parent.parent
    for name in ("options", "stages", "workers"):
        tree = ast.parse((root / package / f"{name}.py").read_text())
        imported = {
            alias.name if isinstance(node, ast.Import) else node.module
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
            for alias in node.names
        }
        foreign = {
            module
            for module in imported
            if module.split(".")[0] not in sys.stdlib_module_names
            and not module.startswith("roboflow_workflows.execution_engine.v2")
        }
        assert foreign == set(), name


def test_passive_pipeline_loads_only_when_used() -> None:
    probe = (
        "import sys\n"
        "import roboflow_workflows.execution_engine.v2\n"
        "from roboflow_workflows.execution_engine.v2.pipelining import PipelineOptions\n"
        "print('pipelining.passive' in ' '.join(sys.modules))\n"
    )
    loaded = subprocess.run(
        [sys.executable, "-B", "-c", probe],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    assert loaded == "False"
