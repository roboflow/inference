from typing import List, Literal, Union

from pydantic import Field

from inference.enterprise.workflows.entities.types import (
    BATCH_OF_IMAGES_KIND,
    BOOLEAN_KIND,
    STRING_KIND,
    InferenceImageSelector,
    InferenceParameterSelector,
    StepOutputSelector,
)
from inference.enterprise.workflows.execution_engine.introspection.entities import (
    ParsedSelector,
    ReferenceDefinition,
    SelectorDefinition,
)
from inference.enterprise.workflows.execution_engine.introspection.selectors_parser import (
    get_step_selectors,
)
from inference.enterprise.workflows.prototypes.block import WorkflowBlockManifest


def test_get_step_selectors_when_no_selectors_defined() -> None:
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        some_integer: int

    step_manifest = Manifest(type="MyManifest", name="my_step", some_integer=3)

    # when
    selectors = get_step_selectors(step_manifest=step_manifest)

    # then
    assert selectors == []


def test_get_step_selectors_when_not_compound_selectors_defined() -> None:
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        image: InferenceImageSelector
        input_parameter: InferenceParameterSelector(
            kind=[BOOLEAN_KIND, STRING_KIND],
        )

    step_manifest = Manifest(
        type="MyManifest",
        name="my_step",
        image="$inputs.image",
        input_parameter="$inputs.param",
    )

    # when
    selectors = sorted(
        get_step_selectors(step_manifest=step_manifest),
        key=lambda s: s.definition.property_name,
    )

    # then
    assert selectors == [
        ParsedSelector(
            definition=SelectorDefinition(
                property_name="image",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="inference_image", kind=[BATCH_OF_IMAGES_KIND]
                    )
                ],
                is_list_element=False,
            ),
            step_name="my_step",
            value="$inputs.image",
            index=None,
        ),
        ParsedSelector(
            definition=SelectorDefinition(
                property_name="input_parameter",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="inference_parameter",
                        kind=[BOOLEAN_KIND, STRING_KIND],
                    )
                ],
                is_list_element=False,
            ),
            step_name="my_step",
            value="$inputs.param",
            index=None,
        ),
    ]


def test_get_step_selectors_when_compound_selectors_defined() -> None:
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        param: List[
            Union[
                InferenceParameterSelector(kind=[BOOLEAN_KIND, STRING_KIND]),
                StepOutputSelector(kind=[STRING_KIND]),
            ]
        ]

    step_manifest = Manifest(
        type="MyManifest", name="my_step", param=["$inputs.param", "$steps.a.param"]
    )

    # when
    selectors = get_step_selectors(step_manifest=step_manifest)

    # then
    assert len(
        selectors
    ), "As list of param values has two elements - two parsed selectors are expected"
    assert (
        selectors[0].value == "$inputs.param"
    ), "First param list value must be shipped first"
    assert selectors[0].index == 0, "First param list value must be shipped first"
    assert (
        selectors[1].value == "$steps.a.param"
    ), "Second param list value must be shipped second"
    assert selectors[1].index == 1, "Second param list value must be shipped second"
    assert (
        selectors[0].definition == selectors[1].definition
    ), "Definitions of selectors must be the same"
    assert (
        selectors[0].definition.property_name == "param"
    ), "Selector definition must hold in terms of property name"
