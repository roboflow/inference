"""Host side: catalogue, compilation for a target and mode, sessions and runs.

The host loads the verified weights once and passes them to every session as
the ``resnet18_state_dict`` resource. Compiling and inspecting never read them.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional

import torch
from classifier import FlipAveragedClassifier
from gallery import Gallery
from redaction import RedactRegion
from roboflow_workflows.execution_engine.v2.blocks import create_catalogue
from roboflow_workflows.execution_engine.v2.catalogue import Catalogue
from roboflow_workflows.execution_engine.v2.compilation import compile_workflow
from roboflow_workflows.execution_engine.v2.plan import (
    CompiledWorkflow,
    CompileOptions,
    RunResult,
)
from roboflow_workflows.execution_engine.v2.targets import Target
from sources import StillFrames

DEMO_DIR = Path(__file__).resolve().parent
WAIT_SECONDS = 60

StateDict = Mapping[str, torch.Tensor]


@dataclass
class DemoContext:
    """What every example receives.

    Attributes:
        state_dict: Verified trained ResNet-18 weights, loaded once.
        gallery: Figures for ``index.html``; disabled without an output dir.
    """

    state_dict: StateDict
    gallery: Gallery


def create_demo_catalogue() -> Catalogue:
    """Merge the built-in V2 catalogue with this demo's block and source.

    Returns:
        Catalogue with built-in blocks, the classifier, the redaction block
        and the still-frame source.
    """
    catalogue = Catalogue.merge(
        create_catalogue(),
        Catalogue([FlipAveragedClassifier, RedactRegion], sources=[StillFrames]),
    )

    return catalogue


def load_definition(name: str) -> Dict[str, Any]:
    """Read one authored workflow from the ``workflows`` directory.

    Args:
        name: File name, for example ``classify.json``.

    Returns:
        The parsed definition.
    """
    definition = json.loads((DEMO_DIR / "workflows" / name).read_text())

    return definition


def compile_for(
    definition: Mapping[str, Any],
    *,
    target: Target = Target.cpu(),
    execution: Literal["run", "phases"] = "run",
    mutation_conflicts: Literal["warn", "error"] = "warn",
) -> CompiledWorkflow:
    """Compile a definition for one target and block execution mode.

    Args:
        definition: Authored workflow.
        target: Capabilities the caller claims for the environment.
        execution: ``run`` calls each block's ``run``; ``phases`` runs the
            selected implementation's phase graph when it has one.
        mutation_conflicts: ``warn`` or ``error`` for unordered in-place
            mutation.

    Returns:
        The compiled plan.
    """
    plan = compile_workflow(
        definition,
        catalogue=create_demo_catalogue(),
        options=CompileOptions(
            target=target,
            block_execution=execution,
            mutation_conflicts=mutation_conflicts,
        ),
    )

    return plan


def run_passive(
    plan: CompiledWorkflow, state_dict: StateDict, inputs: Mapping[str, Any]
) -> RunResult:
    """Create a session (constructs the selected implementations) and run once.

    Args:
        plan: Compiled workflow.
        state_dict: Trained ResNet-18 weights.
        inputs: Workflow inputs.

    Returns:
        The run result, including its trace.
    """
    session = plan.create_session({"resnet18_state_dict": state_dict})
    result = session.run(dict(inputs))

    return result


def output_value(result: Any, name: str) -> Any:
    """Return the raw value of a single-port workflow output.

    Args:
        result: ``RunResult`` or delivered ``GroupResult``.
        name: Workflow output name.

    Returns:
        The value as the engine delivered it (a ``Batch`` for nested axes).
    """
    (entry,) = result.selections[name].values()
    value = result.outputs.data[entry]

    return value


def output_metadata(result: Any, name: str) -> Any:
    """Return the entry metadata of a single-port workflow output.

    Args:
        result: ``RunResult`` or delivered ``GroupResult``.
        name: Workflow output name.

    Returns:
        ``EntryMetadata`` with indexed temporal and source contexts.
    """
    (entry,) = result.selections[name].values()
    metadata = result.outputs.metadata[entry]

    return metadata


def run_active(
    plan: CompiledWorkflow,
    state_dict: StateDict,
    *,
    groups: List[str],
    inputs: Optional[Mapping[str, Any]] = None,
) -> List[Any]:
    """Start a run of a definition with sources, wait for it and collect results.

    Args:
        plan: Compiled workflow with sources.
        state_dict: Trained ResNet-18 weights.
        groups: Output group names to collect.
        inputs: Static workflow inputs.

    Returns:
        Delivered ``GroupResult`` objects in delivery order.

    Raises:
        ActiveRunError: When a step or source fails.
        TimeoutError: When the run does not finish in time.
    """
    session = plan.create_session({"resnet18_state_dict": state_dict})
    delivered: List[Any] = []
    active = session.start(
        dict(inputs or {}),
        handlers={name: delivered.append for name in groups},
    )
    try:
        if not active.wait(timeout=WAIT_SECONDS):
            raise TimeoutError(f"run did not finish within {WAIT_SECONDS} s")
    finally:
        active.stop()

    return delivered
