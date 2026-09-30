"""Host side: load authored JSON, compile it, start a session, collect results.

This is ordinary host code. Compilation, pulses, alignment, windows and
metadata all come from the V2 engine; the host only supplies static inputs,
resources and a handler that records each delivered group result.
"""

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional

from blocks import CelsiusToFahrenheit, ClipEnds, DescribePair, ExposureGuard, GreyCard
from inspection import observe
from probes import FollowerFirst, LeaderAfterFollower
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.errors import ActiveRunError
from sources import CsvSensor, TensorCamera

DEMO_DIR = Path(__file__).resolve().parent
SENSOR_CSV = DEMO_DIR / "data" / "sensor.csv"
WAIT_SECONDS = 20


def create_demo_catalogue() -> Catalogue:
    """Merge the built-in V2 catalogue with this demo's classes.

    Returns:
        Catalogue with built-in blocks and operators, the demo blocks and all
        demo sources, including the two ordering probes.
    """
    catalogue = Catalogue.merge(
        create_catalogue(),
        Catalogue(
            [
                CelsiusToFahrenheit,
                ClipEnds,
                DescribePair,
                ExposureGuard,
                GreyCard,
            ],
            sources=[TensorCamera, CsvSensor, LeaderAfterFollower, FollowerFirst],
        ),
    )

    return catalogue


def load_definition(name: str) -> Dict[str, Any]:
    """Read one authored workflow from the ``workflows`` directory.

    Args:
        name: Relative file name, for example ``rig.json``.

    Returns:
        A fresh, mutable copy of the definition.
    """
    definition = json.loads((DEMO_DIR / "workflows" / name).read_text())

    return definition


def with_changes(definition: Dict[str, Any], **changes: Mapping[str, Any]):
    """Return a copy with literal fields of named sources/operators replaced.

    Args:
        definition: Authored workflow.
        **changes: Declaration name to the fields to set, for example
            ``rig={"missing": "partial"}``.

    Returns:
        Changed copy; the original is untouched.

    Raises:
        KeyError: When a name matches no source or operator.
    """
    changed = copy.deepcopy(definition)
    declarations = changed.get("sources", []) + changed.get("operators", [])
    by_name = {declaration["name"]: declaration for declaration in declarations}
    for name, fields in changes.items():
        by_name[name].update(fields)

    return changed


def group_names(definition: Dict[str, Any]) -> List[str]:
    """List the output groups an authored definition declares.

    Args:
        definition: Authored workflow.

    Returns:
        Names of its ``OutputGroup`` outputs.
    """
    names = [
        output["name"]
        for output in definition["outputs"]
        if output.get("type") == "OutputGroup"
    ]

    return names


@dataclass
class RunRecord:
    """Everything one active run delivered.

    Attributes:
        observations: Observed group results in delivery order.
        plan: The compiled plan that ran.
        active: The finished ``ActiveRun`` (counters, state).
        error: ``ActiveRunError`` raised by ``wait``, if the run failed.
    """

    observations: List[Dict[str, Any]] = field(default_factory=list)
    plan: Any = None
    active: Any = None
    error: Optional[BaseException] = None

    def groups(self, name: str) -> List[Dict[str, Any]]:
        """Return the observations of one group, in delivery order.

        Args:
            name: Output group name.

        Returns:
            Matching observations.
        """
        matching = [record for record in self.observations if record["group"] == name]

        return matching


def run_active(
    definition: Dict[str, Any],
    *,
    inputs: Optional[Mapping[str, Any]] = None,
    resources: Optional[Mapping[str, Any]] = None,
    admission_bound: int = 2,
    on_result: Optional[Callable[[Any, Any], None]] = None,
    expect_failure: bool = False,
    session=None,
) -> RunRecord:
    """Compile (unless a session is given), start, wait and collect one run.

    Args:
        definition: Authored workflow with sources.
        inputs: Static workflow inputs.
        resources: Resources for block/source constructors.
        admission_bound: Pulses one source may have admitted but unprocessed.
        on_result: Optional extra callback ``(result, session)``; the
            session exists even before ``start`` returns its run.
        expect_failure: Keep an ``ActiveRunError`` instead of raising it.
        session: Existing session to start again (restart example).

    Returns:
        Observations, the finished run and its error, if any.
    """
    if session is None:
        plan = compile_workflow(definition, catalogue=create_demo_catalogue())
        session = plan.create_session(dict(resources or {}))
    record = RunRecord(plan=session.plan)

    def receive(result) -> None:
        record.observations.append(observe(result))
        if on_result is not None:
            on_result(result, session)

    record.active = session.start(
        dict(inputs or {}),
        handlers={name: receive for name in group_names(definition)},
        admission_bound=admission_bound,
    )
    try:
        finished = record.active.wait(timeout=WAIT_SECONDS)
        assert finished, f"run did not finish within {WAIT_SECONDS} s"
    except ActiveRunError as error:
        if not expect_failure:
            raise
        record.error = error
    finally:
        record.active.stop()
        if not record.active.done:
            record.active.wait(timeout=WAIT_SECONDS)
    if expect_failure:
        assert record.error is not None, "the run was expected to fail but finished"

    return record
