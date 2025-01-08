import pytest

from inference.core.workflows.core_steps.models.foundation.clip_comparison.v1 import (
    BlockManifest,
)
from inference.core.workflows.errors import RuntimeInputError
from inference.core.workflows.execution_engine.v1.compiler.entities import (
    InputSubstitution,
)
from inference.core.workflows.execution_engine.v1.executor.runtime_input_validator import (
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
                texts="$inputs.text_1",
            ),
            manifest_property="texts",
        ),
        InputSubstitution(
            input_parameter_name="text_2",
            step_manifest=BlockManifest(
                type="ClipComparison",
                name="a",
                image="$inputs.image",
                texts="$inputs.text_2",
            ),
            manifest_property="texts",
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
