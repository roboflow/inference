from inference.enterprise.deployments.complier.utils import (
    get_input_parameters_selectors,
    get_output_names,
    get_output_selectors,
    get_steps_input_selectors,
    get_steps_output_selectors,
    get_steps_selectors,
)
from inference.enterprise.deployments.entities.deployment_specs import DeploymentSpecV1
from inference.enterprise.deployments.errors import (
    DuplicatedSymbolError,
    InvalidReferenceError,
)


def validate_deployment_spec(deployment_spec: DeploymentSpecV1) -> None:
    validate_inputs_names_are_unique(deployment_spec=deployment_spec)
    validate_steps_names_are_unique(deployment_spec=deployment_spec)
    validate_outputs_names_are_unique(deployment_spec=deployment_spec)
    validate_selectors_references_correctness(deployment_spec=deployment_spec)


def validate_inputs_names_are_unique(deployment_spec: DeploymentSpecV1) -> None:
    input_parameters_selectors = get_input_parameters_selectors(
        inputs=deployment_spec.inputs
    )
    if len(input_parameters_selectors) != len(deployment_spec.inputs):
        raise DuplicatedSymbolError("Found duplicated input parameter names")


def validate_steps_names_are_unique(deployment_spec: DeploymentSpecV1) -> None:
    steps_selectors = get_steps_selectors(steps=deployment_spec.steps)
    if len(steps_selectors) != len(deployment_spec.steps):
        raise DuplicatedSymbolError("Found duplicated steps names")


def validate_outputs_names_are_unique(deployment_spec: DeploymentSpecV1) -> None:
    output_names = get_output_names(outputs=deployment_spec.outputs)
    if len(output_names) != len(deployment_spec.outputs):
        raise DuplicatedSymbolError("Found duplicated outputs names")


def validate_selectors_references_correctness(
    deployment_spec: DeploymentSpecV1,
) -> None:
    input_parameters_selectors = get_input_parameters_selectors(
        inputs=deployment_spec.inputs
    )
    steps_inputs_selectors = get_steps_input_selectors(steps=deployment_spec.steps)
    steps_output_selectors = get_steps_output_selectors(steps=deployment_spec.steps)
    output_selectors = get_output_selectors(outputs=deployment_spec.outputs)
    all_possible_input_selectors = input_parameters_selectors | steps_output_selectors
    for step_input_selector in steps_inputs_selectors:
        if step_input_selector not in all_possible_input_selectors:
            raise InvalidReferenceError(
                f"Detected step input selector: {step_input_selector} that is not defined as valid input."
            )
    for output_selector in output_selectors:
        if output_selector not in steps_output_selectors:
            raise InvalidReferenceError(
                f"Detected output selector: {output_selector} that is not defined as valid output of any of the steps."
            )
