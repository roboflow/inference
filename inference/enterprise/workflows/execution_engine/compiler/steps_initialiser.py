from typing import Any, Callable, Dict, List

from inference.enterprise.workflows.execution_engine.compiler.entities import (
    BlockSpecification,
    InitialisedStep,
)
from inference.enterprise.workflows.prototypes.block import WorkflowBlockManifest


def initialise_steps(
    steps_manifest: List[WorkflowBlockManifest],
    available_bocks: List[BlockSpecification],
    explicit_init_parameters: Dict[str, Any],
    initializers: Dict[str, Callable[[None], Any]],
) -> List[InitialisedStep]:
    available_bocks_by_manifest_class = {
        block.manifest_class: block for block in available_bocks
    }
    initialised_steps = []
    for step_manifest in steps_manifest:
        if type(step_manifest) not in available_bocks_by_manifest_class:
            raise ValueError(
                f"Provided step manifest of type ({type(step_manifest)}) "
                f"which is not compatible with registered workflow blocks."
            )
        block_specification = available_bocks_by_manifest_class[type(step_manifest)]
        initialised_step = initialise_step(
            step_manifest=step_manifest,
            block_specification=block_specification,
            explicit_init_parameters=explicit_init_parameters,
            initializers=initializers,
        )
        initialised_steps.append(initialised_step)
    return initialised_steps


def initialise_step(
    step_manifest: WorkflowBlockManifest,
    block_specification: BlockSpecification,
    explicit_init_parameters: Dict[str, Any],
    initializers: Dict[str, Callable[[None], Any]],
) -> InitialisedStep:
    block_init_parameters = block_specification.block_class.get_init_parameters()
    init_parameters_values = retrieve_init_parameters_values(
        block_init_parameters=block_init_parameters,
        block_source=block_specification.block_source,
        explicit_init_parameters=explicit_init_parameters,
        initializers=initializers,
    )
    step = block_specification.block_class(**init_parameters_values)
    return InitialisedStep(
        block_specification=block_specification,
        manifest=step_manifest,
        step=step,
    )


def retrieve_init_parameters_values(
    block_init_parameters: List[str],
    block_source: str,
    explicit_init_parameters: Dict[str, Any],
    initializers: Dict[str, Callable[[None], Any]],
) -> Dict[str, Any]:
    return {
        block_init_parameter: retrieve_init_parameter_values(
            block_init_parameter=block_init_parameter,
            block_source=block_source,
            explicit_init_parameters=explicit_init_parameters,
            initializers=initializers,
        )
        for block_init_parameter in block_init_parameters
    }


def retrieve_init_parameter_values(
    block_init_parameter: str,
    block_source: str,
    explicit_init_parameters: Dict[str, Any],
    initializers: Dict[str, Callable[[None], Any]],
) -> Any:
    full_parameter_name = f"{block_source}.{block_init_parameter}"
    if full_parameter_name in explicit_init_parameters:
        return explicit_init_parameters[full_parameter_name]
    if full_parameter_name in initializers:
        return initializers[full_parameter_name]
    if block_init_parameter in explicit_init_parameters:
        return explicit_init_parameters[block_init_parameter]
    raise ValueError(
        f"Could not resolve init parameter {block_init_parameter} to initialise "
        f"step from plugin: {block_source}."
    )
