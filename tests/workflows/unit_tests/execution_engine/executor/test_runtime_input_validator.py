import pytest

from inference.enterprise.workflows.core_steps.models.foundation.clip_comparison import (
    BlockManifest,
)
from inference.enterprise.workflows.errors import RuntimeInputError
from inference.enterprise.workflows.execution_engine.compiler.entities import (
    InputSubstitution,
)
from inference.enterprise.workflows.execution_engine.executor.runtime_input_validator import (
    validate_runtime_input,
)


def test_validate_runtime_input_when_input_is_valid() -> None:
    # given
    input_substitutions = [
        InputSubstitution(
            input_parameter_name="text_1",
            step_manifest=BlockManifest(
                type="ClipComparison",
                name="a",
                image="$inputs.image",
                text="$inputs.text_1",
            ),
            manifest_property="text",
        ),
        InputSubstitution(
            input_parameter_name="text_2",
            step_manifest=BlockManifest(
                type="ClipComparison",
                name="a",
                image="$inputs.image",
                text="$inputs.text_2",
            ),
            manifest_property="text",
        ),
    ]

    # when
    validate_runtime_input(
        runtime_parameters={
            "text_1": ["some", "other"],
            "text_2": ["also", "a", "list"],
        },
        input_substitutions=input_substitutions,
    )

    # then - no error


def test_validate_runtime_input_when_input_is_invalid() -> None:
    # given
    input_substitutions = [
        InputSubstitution(
            input_parameter_name="text_1",
            step_manifest=BlockManifest(
                type="ClipComparison",
                name="a",
                images="$inputs.image",
                texts="$inputs.text_1",
            ),
            manifest_property="texts",
        )
    ]

    # when
    with pytest.raises(RuntimeInputError):
        validate_runtime_input(
            runtime_parameters={
                "text_1": "should_be_a_list_of_str",
            },
            input_substitutions=input_substitutions,
        )
