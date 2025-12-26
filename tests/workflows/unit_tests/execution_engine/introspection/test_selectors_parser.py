from typing import List, Literal, Union

from pydantic import Field

from inference.core.workflows.execution_engine.entities.base import OutputDefinition
from inference.core.workflows.execution_engine.entities.types import (
    BOOLEAN_KIND,
    IMAGE_KIND,
    LIST_OF_VALUES_KIND,
    STRING_KIND,
    Selector,
    StepOutputSelector,
    WorkflowImageSelector,
    WorkflowParameterSelector,
)
from inference.core.workflows.execution_engine.introspection.entities import (
    ParsedSelector,
    ReferenceDefinition,
    SelectorDefinition,
)
from inference.core.workflows.execution_engine.introspection.selectors_parser import (
    get_step_selectors,
)
from inference.core.workflows.prototypes.block import WorkflowBlockManifest


def test_get_step_selectors_when_no_selectors_defined() -> None:
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        some_integer: int

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

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
        image: WorkflowImageSelector
        input_parameter: WorkflowParameterSelector(
            kind=[BOOLEAN_KIND, STRING_KIND],
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

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
                        selected_element="workflow_image",
                        kind=[IMAGE_KIND],
                        points_to_batch={True},
                    )
                ],
                is_list_element=False,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            ),
            step_name="my_step",
            value="$inputs.image",
            index=None,
            key=None,
        ),
        ParsedSelector(
            definition=SelectorDefinition(
                property_name="input_parameter",
                property_description="not available",
                allowed_references=[
                    ReferenceDefinition(
                        selected_element="workflow_parameter",
                        kind=[BOOLEAN_KIND, STRING_KIND],
                        points_to_batch={False},
                    )
                ],
                is_list_element=False,
                is_dict_element=False,
                dimensionality_offset=0,
                is_dimensionality_reference_property=False,
            ),
            step_name="my_step",
            value="$inputs.param",
            index=None,
            key=None,
        ),
    ]


def test_get_step_selectors_when_compound_selectors_defined() -> None:
    # given

    class Manifest(WorkflowBlockManifest):
        type: Literal["MyManifest"]
        name: str = Field(description="name field")
        param: List[
            Union[
                WorkflowParameterSelector(kind=[BOOLEAN_KIND, STRING_KIND]),
                StepOutputSelector(kind=[STRING_KIND]),
            ]
        ]

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

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


def test_get_step_selectors_when_union_of_list_str_or_selector_receives_selector() -> (
    None
):
    # Union[List[str], Selector] with a selector value
    class Manifest(WorkflowBlockManifest):
        type: Literal["UnionListTest"]
        name: str = Field(description="name field")
        tags: Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])] = Field(
            description="Tags can be a literal list or a selector to a list"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    step_manifest = Manifest(type="UnionListTest", name="my_step", tags="$inputs.tags")
    selectors = get_step_selectors(step_manifest=step_manifest)

    assert len(selectors) == 1
    assert selectors[0].value == "$inputs.tags"
    assert selectors[0].definition.is_list_element is False
    assert selectors[0].index is None


def test_get_step_selectors_when_union_of_list_str_or_selector_receives_literal_list() -> (
    None
):
    # Union[List[str], Selector] with a literal list - no selectors
    class Manifest(WorkflowBlockManifest):
        type: Literal["UnionListTest"]
        name: str = Field(description="name field")
        tags: Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])] = Field(
            description="Tags can be a literal list or a selector to a list"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    step_manifest = Manifest(
        type="UnionListTest", name="my_step", tags=["tag1", "tag2", "tag3"]
    )
    selectors = get_step_selectors(step_manifest=step_manifest)

    assert len(selectors) == 0


def test_get_step_selectors_when_union_of_list_str_or_selector_receives_list_with_selector_like_strings() -> (
    None
):
    # List[str] should NOT parse selector-like strings - they're just strings
    class Manifest(WorkflowBlockManifest):
        type: Literal["UnionListTest"]
        name: str = Field(description="name field")
        tags: Union[List[str], Selector(kind=[LIST_OF_VALUES_KIND])] = Field(
            description="Tags can be a literal list or a selector to a list"
        )

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    step_manifest = Manifest(
        type="UnionListTest",
        name="my_step",
        tags=["literal_tag", "$inputs.tag", "$inputs.another_tag"],
    )
    selectors = get_step_selectors(step_manifest=step_manifest)

    # these look like selectors but List[str] means they're literal strings
    assert len(selectors) == 0


def test_get_step_selectors_when_union_of_list_with_selectors_or_selector_receives_mixed_list() -> (
    None
):
    # List[Union[Selector, str]] SHOULD parse selectors inside the list
    class Manifest(WorkflowBlockManifest):
        type: Literal["UnionListTest"]
        name: str = Field(description="name field")
        registration_tags: Union[
            List[Union[Selector(kind=[STRING_KIND]), str]],
            Selector(kind=[LIST_OF_VALUES_KIND]),
        ] = Field(description="Tags with selectors inside the list")

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    step_manifest = Manifest(
        type="UnionListTest",
        name="my_step",
        registration_tags=["literal_tag", "$inputs.tag", "$inputs.another_tag"],
    )
    selectors = get_step_selectors(step_manifest=step_manifest)

    assert len(selectors) == 2
    assert selectors[0].value == "$inputs.tag"
    assert selectors[0].index == 1
    assert selectors[1].value == "$inputs.another_tag"
    assert selectors[1].index == 2


def test_get_step_selectors_when_union_of_list_with_selectors_or_selector_receives_direct_selector() -> (
    None
):
    # Union[List[Union[Selector, str]], Selector] with a direct selector
    class Manifest(WorkflowBlockManifest):
        type: Literal["UnionListTest"]
        name: str = Field(description="name field")
        registration_tags: Union[
            List[Union[Selector(kind=[STRING_KIND]), str]],
            Selector(kind=[LIST_OF_VALUES_KIND]),
        ] = Field(description="Tags with selectors inside the list")

        @classmethod
        def describe_outputs(cls) -> List[OutputDefinition]:
            return []

    step_manifest = Manifest(
        type="UnionListTest", name="my_step", registration_tags="$inputs.tags"
    )
    selectors = get_step_selectors(step_manifest=step_manifest)

    assert len(selectors) == 1
    assert selectors[0].value == "$inputs.tags"
    assert selectors[0].index is None
