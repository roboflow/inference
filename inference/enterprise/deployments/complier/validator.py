from inference.enterprise.deployments.complier.utils import (
    get_input_parameters_selectors,
    get_steps_input_selectors,
    get_steps_output_selectors,
    get_output_selectors,
    get_selectors_from_condition_specs,
    get_steps_selectors,
    get_output_names,
)
from inference.enterprise.deployments.entities.deployment_specs import DeploymentSpecV1
from inference.enterprise.deployments.entities.steps import CVModel, Crop, Condition
from inference.enterprise.deployments.errors import (
    InvalidReferenceError,
    VariableNotBounderError,
    DuplicatedSymbolError,
)


def validate_deployment_spec(deployment_spec: DeploymentSpecV1) -> None:
    validate_inputs_names_are_unique(deployment_spec=deployment_spec)
    validate_steps_names_are_unique(deployment_spec=deployment_spec)
    validate_outputs_names_are_unique(deployment_spec=deployment_spec)
    validate_selectors_references(deployment_spec=deployment_spec)
    validate_steps_required_inputs(deployment_spec=deployment_spec)


def validate_inputs_names_are_unique(deployment_spec: DeploymentSpecV1) -> None:
    input_parameters_selectors = get_input_parameters_selectors(
        deployment_spec=deployment_spec
    )
    if len(input_parameters_selectors) != len(deployment_spec.inputs):
        raise DuplicatedSymbolError("Found duplicated input parameter names")


def validate_steps_names_are_unique(deployment_spec: DeploymentSpecV1) -> None:
    steps_selectors = get_steps_selectors(deployment_spec=deployment_spec)
    if len(steps_selectors) != len(deployment_spec.steps):
        raise DuplicatedSymbolError("Found duplicated steps names")


def validate_outputs_names_are_unique(deployment_spec: DeploymentSpecV1) -> None:
    output_names = get_output_names(deployment_spec=deployment_spec)
    if len(output_names) != len(deployment_spec.outputs):
        raise DuplicatedSymbolError("Found duplicated outputs names")


def validate_selectors_references(deployment_spec: DeploymentSpecV1) -> None:
    input_parameters_selectors = get_input_parameters_selectors(
        deployment_spec=deployment_spec
    )
    steps_inputs_selectors = get_steps_input_selectors(deployment_spec=deployment_spec)
    steps_output_selectors = get_steps_output_selectors(deployment_spec=deployment_spec)
    output_selectors = get_output_selectors(deployment_spec=deployment_spec)
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


def validate_steps_required_inputs(deployment_spec: DeploymentSpecV1) -> None:
    for step in deployment_spec.steps:
        if step.type not in STEP_TYPE2INPUT_VALIDATOR:
            continue
        STEP_TYPE2INPUT_VALIDATOR[step.type](step)


def validate_model_step_inputs(model_step: CVModel) -> None:
    input_names = set(model_step.inputs.keys())
    if "image" not in input_names:
        raise VariableNotBounderError(
            f"Required input `image` not defined for CVModel step: {model_step.name}"
        )


def validate_crop_inputs(crop_step: Crop) -> None:
    input_names = set(crop_step.inputs.keys())
    if "image" not in input_names:
        raise VariableNotBounderError(
            f"Required input `image` not defined for Crop step: {crop_step.name}"
        )
    if "predictions" not in input_names:
        raise VariableNotBounderError(
            f"Required input `predictions` not defined for Crop step: {crop_step.name}"
        )


def validate_condition_inputs(condition_step: Condition) -> None:
    input_names = set(condition_step.inputs.keys())
    input_names_as_selectors = {f"${name}" for name in input_names}
    condition_selectors = get_selectors_from_condition_specs(
        condition_specs=condition_step.condition
    )
    unbounded_variables = input_names_as_selectors.symmetric_difference(
        condition_selectors
    )
    if len(unbounded_variables) != 0:
        raise VariableNotBounderError(
            f"Detected unbounded variables: {unbounded_variables} in Condition step: {condition_step.name}"
        )


STEP_TYPE2INPUT_VALIDATOR = {
    "CVModel": validate_model_step_inputs,
    "Crop": validate_crop_inputs,
    "Condition": validate_condition_inputs,
}
